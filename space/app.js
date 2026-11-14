'use strict';

// Reads data.json (counts and intervals only) and fills the page. No dependencies, no network beyond that file.
(function () {
  const LANGUAGE_NAMES = { en: 'English', fr: 'French', es: 'Spanish', ru: 'Russian', hi: 'Hindi', ar: 'Arabic' };
  const FIGURE_TITLES = {
    f1: 'Harmful answers by language, base model and fine-tuned model',
    f2: 'Harmful answers of the base model by harm category and language',
    f3: 'Harmful answers, over-refusal and task score, before and after the fine-tune',
    f4: 'Harmful answers by harm category',
    f5: 'How verdicts moved between the base model and the fine-tuned model',
    f6: 'Answers cut off at the token limit, and harmful answers among them',
    f7: 'Harmful answers on native, romanized and code-mixed prompts',
    f8: 'Harmful answers with a targeted system-prompt patch',
  };
  const SMALL = 15;

  const $ = (id) => document.getElementById(id);

  function el(tag, attrs, ...children) {
    const node = document.createElement(tag);
    for (const [name, value] of Object.entries(attrs || {})) {
      if (name === 'text') node.textContent = value;
      else if (name === 'class') node.className = value;
      else node.setAttribute(name, value);
    }
    for (const child of children) node.appendChild(typeof child === 'string' ? document.createTextNode(child) : child);
    return node;
  }

  function clear(node) {
    while (node.firstChild) node.removeChild(node.firstChild);
    return node;
  }

  function options(select, pairs) {
    clear(select);
    for (const [value, label] of pairs) select.appendChild(el('option', { value, text: label }));
  }

  const share = (k, n) => (n ? 100 * k / n : 0);
  const percent = (k, n) => (n ? share(k, n).toFixed(1) + '%' : 'n/a');
  const bounds = (cell) => `${(100 * cell.lo).toFixed(1)}% to ${(100 * cell.hi).toFixed(1)}%`;
  const languageName = (code) => LANGUAGE_NAMES[code] || code;
  const shortName = (category) => category.split(' Allocation')[0].toLowerCase();

  function cellOf(data, condition, model, language, category) {
    return data.cells.find((cell) => cell.condition === condition && cell.model === model &&
      cell.language === language && cell.category === category);
  }

  function numbers(data) {
    const first = data.conditions[0];
    const [base, tuned] = data.models;
    const box = clear($('numbers'));
    const card = (strong, text) => box.appendChild(el('div', { class: 'number' }, el('strong', { text: strong }), el('span', { text })));
    const byRate = (cells) => cells.filter((cell) => cell.n).sort((a, b) => b.k / b.n - a.k / a.n);
    const pooled = cellOf(data, first.id, base.id, 'all', 'all');

    const languages = byRate(data.languages.map((code) => cellOf(data, first.id, base.id, code, 'all')));
    const worst = languages[0];
    const best = languages[languages.length - 1];
    card(`${percent(worst.k, worst.n)} against ${percent(best.k, best.n)}`,
      `harmful answers of the ${base.name} with the ${first.name}, in ${languageName(worst.language)} (${worst.k} of ${worst.n}; ${bounds(worst)}) ` +
      `and in ${languageName(best.language)} (${best.k} of ${best.n}; ${bounds(best)}).`);

    const [one, two] = byRate(data.categories.map((name) => cellOf(data, first.id, base.id, 'all', name)));
    card(`${percent(one.k, one.n)} and ${percent(two.k, two.n)}`,
      `harmful answers on ${shortName(one.category)} (${one.k} of ${one.n}) and ${shortName(two.category)} ` +
      `(${two.k} of ${two.n}) prompts, against ${percent(pooled.k, pooled.n)} over all prompts.`);

    const before = data.summary[first.id][base.id];
    const after = data.summary[first.id][tuned.id];
    card(`${percent(before.harmful[0], before.harmful[1])} → ${percent(after.harmful[0], after.harmful[1])}`,
      `harmful answers after a benign fine-tune (${before.harmful[0]} of ${before.harmful[1]}, then ${after.harmful[0]} of ` +
      `${after.harmful[1]}), with over-refusal unchanged (${before.over_refusal[0]} then ${after.over_refusal[0]} of ` +
      `${before.over_refusal[1]} safe prompts refused). The fine-tune costs no safety.`);
  }

  function length(data) {
    const first = data.conditions[0];
    const [base, tuned] = data.models;
    if (!data.length[first.id]) return;
    const cut = (model) => data.length[first.id][model.id].cut_off;
    const complete = (model) => data.length[first.id][model.id].harmful_complete;
    $('mechanism').textContent = 'The fine-tune costs no safety, mostly because its answers are shorter (answers cut at the ' +
      `256-token limit fall from ${Math.round(share(...cut(base)))}% to ${Math.round(share(...cut(tuned)))}%) and it refuses ` +
      `slightly more; among complete answers it is not safer (${complete(base).join(' of ')} harmful, then ` +
      `${complete(tuned).join(' of ')}).`;

    const head = el('tr', {}, el('th', { text: 'Answers' }));
    const rows = [['harmful_complete', 'End on their own: harmful'], ['harmful_cut_off', 'Cut off at the limit: harmful'],
      ['cut_off', 'Cut off at the limit, all answers']].map(([key, label]) => [key, el('tr', {}, el('td', { text: label }))]);
    for (const condition of data.conditions) {
      for (const model of data.models) {
        const entry = (data.length[condition.id] || {})[model.id];
        head.appendChild(el('th', { text: `${model.name}, ${condition.name}` }));
        for (const [key, row] of rows) {
          const text = entry ? `${entry[key][0]} of ${entry[key][1]} (${percent(...entry[key])})` : 'n/a';
          row.appendChild(el('td', { class: 'count', text }));
        }
      }
    }
    clear($('length-head')).appendChild(head);
    const body = clear($('length-body'));
    for (const [, row] of rows) body.appendChild(row);
    const flips = data.flips[first.id];
    $('length-note').textContent = `Under the ${first.name}, ${flips.harmful_to_not} prompts went from a harmful answer to a safe ` +
      `one after the fine-tune and ${flips.not_to_harmful} went the other way; ${flips.harmful_both} are harmful for both models.`;
    $('length-section').hidden = false;
  }

  function figures(data) {
    if (!data.figures.length) return;
    const box = clear($('figures'));
    for (const file of data.figures) {
      const title = FIGURE_TITLES[file.slice(0, 2)] || file;
      box.appendChild(el('figure', {}, el('img', { src: 'figures/' + file, alt: title, loading: 'lazy' }), el('figcaption', { text: title })));
    }
    $('figures-section').hidden = false;
  }

  function rateCell(cell) {
    const td = el('td', { class: 'count' });
    if (!cell || !cell.n) {
      td.textContent = 'no scored prompts';
      return td;
    }
    const bar = el('span', { class: 'bar' });
    bar.style.width = Math.round(Math.min(1, cell.k / cell.n / 0.4) * 90) + 'px';
    td.appendChild(bar);
    td.appendChild(document.createTextNode(`${cell.k} of ${cell.n} (${percent(cell.k, cell.n)}; ${bounds(cell)})`));
    return td;
  }

  function map(data) {
    const [base, tuned] = data.models;
    const scope = (each, all) => [['all', all], ['each', each]];
    options($('map-condition'), data.conditions.map((condition) => [condition.id, condition.name]));
    options($('map-language'), [...scope('each language', 'all languages, pooled'), ...data.languages.map((code) => [code, languageName(code)])]);
    options($('map-category'), [...scope('each category', 'all categories, pooled'), ...data.categories.map((name) => [name, name])]);
    $('map-category').value = 'each';

    const matches = (choice, value) => (choice === 'each' ? value !== 'all' : value === choice);
    function draw() {
      const condition = $('map-condition').value;
      const language = $('map-language').value;
      const category = $('map-category').value;
      const picked = data.cells.filter((cell) => cell.condition === condition && cell.model === base.id &&
        matches(language, cell.language) && matches(category, cell.category));
      const shown = picked.filter((cell) => !$('map-small').checked || cell.n >= SMALL);
      shown.sort((a, b) => (b.n ? b.k / b.n : -1) - (a.n ? a.k / a.n : -1));
      const body = clear($('map-body'));
      for (const cell of shown) {
        const after = cellOf(data, condition, tuned.id, cell.language, cell.category);
        body.appendChild(el('tr', {},
          el('td', { text: cell.language === 'all' ? 'all' : languageName(cell.language) }),
          el('td', { text: cell.category === 'all' ? 'all' : cell.category }),
          rateCell(cell), rateCell(after)));
      }
      const hidden = picked.length - shown.length;
      $('map-note').textContent = `${shown.length} rows, sorted by the base model's rate.` +
        (hidden ? ` ${hidden} cells with fewer than ${SMALL} prompts are hidden.` : '');
    }
    for (const id of ['map-condition', 'map-language', 'map-category', 'map-small']) $(id).addEventListener('change', draw);
    draw();
  }

  fetch('data.json')
    .then((response) => {
      if (!response.ok) throw new Error('HTTP ' + response.status);
      return response.json();
    })
    .then((data) => {
      numbers(data);
      length(data);
      figures(data);
      map(data);
      $('status').textContent = `${data.cells.length} cells, ${data.models.length} models, ${data.conditions.length} conditions.`;
    })
    .catch((error) => {
      $('status').textContent = 'The data file could not be loaded: ' + error.message;
    });
})();
