'use strict';

// The replay explorer: reads explorer.json after the page has painted and shows, for one prompt at a time, what
// each measured system answered. Nothing runs a model; every answer was stored when the evaluation ran.
(function () {
  const { el, clear, options, languageName } = window.pageKit;
  const BADGES = {
    refused: ['refused', 'Refused'],
    answered: ['answered', 'Answered harmlessly'],
    harmful: ['harmful', 'Harmful'],
    unscored: ['unscored', 'No verdict'],
  };
  const KINDS = { romanized: 'in Latin script', codemixed: 'mixed with English' };
  const ALL = 'all';
  const $ = (id) => document.getElementById(id);
  const guarded = (answer) => answer.verdict === 'harmful' || answer.verdict === 'unscored';
  const scored = (answer) => answer && answer.verdict !== 'unscored';
  const isHarmful = (answer) => answer.verdict === 'harmful';

  let data = null;
  let shown = [];

  // every answer a prompt has, its own and those of its script variants
  function answersOf(prompt) {
    const found = Object.values(prompt.answers);
    for (const variant of prompt.variants) found.push(...Object.values(variant.answers));
    return found.filter(scored);
  }

  function filters() {
    const [first, ...others] = data.systems;
    const list = [
      [ALL, 'all prompts', () => true],
      ['disagree', 'prompts where the systems disagree', (prompt) => prompt.disagree],
      ['every', 'prompts every system answers harmfully', (prompt) => {
        const found = answersOf(prompt);
        return found.length > 0 && found.every(isHarmful);
      }],
    ];
    const pair = (prompt, other) => [prompt.answers[first.id], prompt.answers[other.id]];
    for (const other of others) {
      list.push([`gone:${other.id}`, `harmful for: ${first.name}; not for: ${other.name}`, (prompt) => {
        const [before, after] = pair(prompt, other);
        return scored(before) && scored(after) && isHarmful(before) && !isHarmful(after);
      }]);
      list.push([`new:${other.id}`, `not harmful for: ${first.name}; harmful for: ${other.name}`, (prompt) => {
        const [before, after] = pair(prompt, other);
        return scored(before) && scored(after) && !isHarmful(before) && isHarmful(after);
      }]);
    }
    if (data.prompts.some((prompt) => prompt.variants.length)) {
      list.push(['variant', 'a script variant changes the verdict', (prompt) => prompt.variants.some((variant) =>
        Object.entries(variant.answers).some(([system, answer]) => scored(answer) && scored(prompt.answers[system]) &&
          isHarmful(answer) !== isHarmful(prompt.answers[system])))]);
    }
    return list;
  }

  function card(system, answer, code) {
    const [kind, label] = BADGES[answer.verdict];
    const box = el('div', { class: 'system' },
      el('p', { class: 'system-name' }, el('strong', { text: system.name })),
      el('p', { class: 'note', text: system.note ? `${system.condition} · ${system.note}` : system.condition }),
      el('p', {},
        el('span', { class: `badge ${kind}`, text: label }),
        el('span', { class: 'tag', text: answer.cut_off ? 'hit the 256-token limit' : 'ended on its own' })));
    const body = el('div', { class: 'answer-box' });
    body.hidden = true;
    const toggle = el('button', { type: 'button', 'aria-expanded': 'false', text: 'Show answer' });
    const text = () => {
      if (answer.text === null) {
        const why = answer.verdict === 'harmful' ? 'The text of answers judged harmful is not shown here' :
          'This answer has no verdict, so its text is not shown here';
        return el('p', { class: 'note', text: `${why}; it is in the responses dataset.` });
      }
      const quote = el('blockquote', { class: 'answer', lang: code, dir: 'auto', text: answer.text });
      const cut = answer.clipped ? el('p', { class: 'note', text: `Cut to its first ${answer.text.length} characters.` }) : null;
      return cut ? el('div', {}, quote, cut) : quote;
    };
    const fill = () => {
      clear(body);
      if (!guarded(answer) || answer.text === null) {
        body.appendChild(text());
        return;
      }
      const reason = answer.verdict === 'harmful' ? 'The judge called this answer harmful.' :
        'The judge gave no verdict on this answer, so it is treated as harmful.';
      const confirm = el('button', { type: 'button', text: `Show the first ${data.cut} characters` });
      confirm.addEventListener('click', () => {
        clear(body).appendChild(text());
      });
      body.appendChild(el('p', { class: 'warning' }, el('strong', { text: 'Content warning. ' }),
        `${reason} Only its first ${data.cut} characters are kept here.`));
      body.appendChild(confirm);
    };
    toggle.addEventListener('click', () => {
      body.hidden = !body.hidden;
      toggle.setAttribute('aria-expanded', String(!body.hidden));
      toggle.textContent = body.hidden ? 'Show answer' : 'Hide answer';
      if (!body.hidden) fill();
    });
    box.appendChild(toggle);
    box.appendChild(body);
    return box;
  }

  function cards(answers, code) {
    const grid = el('div', { class: 'systems' });
    for (const system of data.systems) {
      if (answers[system.id]) grid.appendChild(card(system, answers[system.id], code));
    }
    return grid;
  }

  function link(prompt) {
    return `#${$('ex-language').value}/${encodeURIComponent($('ex-category').value)}/${prompt.id}`;
  }

  function show(prompt) {
    const panel = clear($('ex-panel'));
    if (!prompt) return;
    const code = prompt.language;
    panel.appendChild(el('p', { class: 'note', text: `${languageName(code)} · ${prompt.categories.join(' · ')}` }));
    panel.appendChild(el('blockquote', { class: 'prompt', lang: code, dir: 'auto', text: prompt.text }));
    panel.appendChild(el('p', { class: 'note' }, el('a', { href: link(prompt), text: 'Link to this prompt' })));
    panel.appendChild(cards(prompt.answers, code));
    for (const variant of prompt.variants) {
      const kind = KINDS[variant.kind] || variant.kind;
      panel.appendChild(el('h3', { text: `Script variant: ${languageName(code)} ${kind}` }));
      panel.appendChild(el('blockquote', { class: 'prompt', lang: code, dir: 'auto', text: variant.text }));
      panel.appendChild(cards(variant.answers, code));
    }
  }

  // keep the address on the prompt a visitor chose, so it can be copied as a link
  function mark() {
    const prompt = current();
    if (prompt) history.replaceState(null, '', link(prompt));
  }

  function current() {
    return shown.find((prompt) => prompt.id === $('ex-prompt').value);
  }

  // rebuild the prompt list for the three filters; keep the chosen prompt when it is still in the list
  function list(keep) {
    const language = $('ex-language').value;
    const category = $('ex-category').value;
    const test = filters().find(([id]) => id === $('ex-filter').value)[2];
    shown = data.prompts.filter((prompt) => (language === ALL || prompt.language === language) &&
      (category === ALL || prompt.categories.includes(category)) && test(prompt));
    const short = (text) => (text.length > 80 ? text.slice(0, 80) + '…' : text);
    options($('ex-prompt'), shown.map((prompt, index) => [prompt.id, `${index + 1}. ${short(prompt.text)}`]));
    if (keep && shown.some((prompt) => prompt.id === keep)) $('ex-prompt').value = keep;
    $('ex-count').textContent = shown.length ? `${shown.length} prompts match.` : 'No prompt matches these filters.';
    for (const id of ['ex-previous', 'ex-next']) $(id).disabled = shown.length < 2;
    show(current());
  }

  function step(by) {
    if (!shown.length) return;
    const at = shown.findIndex((prompt) => prompt.id === $('ex-prompt').value);
    $('ex-prompt').value = shown[(at + by + shown.length) % shown.length].id;
    show(current());
    mark();
  }

  // #language/category/prompt-id: open that prompt, widening the filters when it is not among their matches
  function follow() {
    const parts = location.hash.slice(1).split('/');
    if (parts.length !== 3) return false;
    const prompt = data.prompts.find((row) => row.id === parts[2]);
    if (!prompt) return false;
    const category = decodeURIComponent(parts[1]);
    $('ex-language').value = parts[0] === prompt.language ? parts[0] : ALL;
    $('ex-category').value = prompt.categories.includes(category) ? category : ALL;
    $('ex-filter').value = ALL;
    list(prompt.id);
    return true;
  }

  function start(loaded) {
    data = loaded;
    const categories = [...new Set(data.prompts.flatMap((prompt) => prompt.categories))].sort();
    const languages = [...new Set(data.prompts.map((prompt) => prompt.language))];
    options($('ex-language'), [[ALL, 'all languages'], ...languages.map((code) => [code, languageName(code)])]);
    options($('ex-category'), [[ALL, 'all categories'], ...categories.map((name) => [name, name])]);
    options($('ex-filter'), filters().map(([id, label]) => [id, label]));
    $('ex-filter').value = 'disagree';
    $('ex-scope').textContent = `${data.prompts.length} of the ${data.total} harmful-set prompts are here, taken in ` +
      `this order: prompts where the systems disagree on a harmful answer (${data.disagreeing}), prompts every system ` +
      `answers harmfully (${data.persistent}), then prompts spread evenly over the languages. The ` +
      `${data.left_out.prompts} prompts of the category "${data.left_out.category}" are left out. ` +
      `Systems: ${data.systems.map((system) => `${system.name} (${system.condition})`).join('; ')}.`;
    $('ex-harmful').textContent = data.harmful_text ?
      `An answer the judge called harmful is cut to its first ${data.cut} characters and needs a second click.` :
      'The text of answers judged harmful is not shown here; it is in the responses dataset.';
    for (const id of ['ex-language', 'ex-category', 'ex-filter']) {
      $(id).addEventListener('change', () => {
        list(current() && current().id);
        mark();
      });
    }
    $('ex-prompt').addEventListener('change', () => {
      show(current());
      mark();
    });
    $('ex-previous').addEventListener('click', () => step(-1));
    $('ex-next').addEventListener('click', () => step(1));
    window.addEventListener('hashchange', follow);
    $('explorer-section').hidden = false;
    if (follow()) $('explorer-section').scrollIntoView();
    else list(null);
  }

  function load() {
    fetch('explorer.json')
      .then((response) => (response.ok ? response.json() : Promise.reject(new Error('HTTP ' + response.status))))
      .then(start)
      // without the file the page keeps its tables and the section stays hidden
      .catch(() => {});
  }

  // a link to a prompt loads the file at once; otherwise it waits until the page has painted
  if (location.hash.split('/').length === 3) load();
  else if (window.requestIdleCallback) window.requestIdleCallback(load);
  else window.setTimeout(load, 200);
})();
