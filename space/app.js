'use strict';

// Reads data.json (counts and labels only) and fills the page. No dependencies, no network beyond that file.
(function () {
  const LANGUAGE_NAMES = { en: 'English', fr: 'French', es: 'Spanish', ru: 'Russian', hi: 'Hindi', ar: 'Arabic' };
  const FIGURE_TITLES = {
    f1: 'Safe rate by language, base model and fine-tuned model',
    f2: 'Harmful answers of the base model by language and harm category',
    f3: 'Harmful answers on native, romanized and code-mixed prompts',
    f4: 'Harmful answers, over-refusal and task score, before and after',
    f5: 'The fine-tuned model at q4_k_m',
  };
  const SMALL = 15;
  const PAGE = 50;

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

  const percent = (k, n) => (n ? (100 * k / n).toFixed(1) + '%' : 'n/a');
  const languageName = (code) => LANGUAGE_NAMES[code] || code;
  const pValue = (p) => (p < 0.001 ? 'p < 0.001' : 'p = ' + p.toFixed(3));

  function cellOf(data, condition, model, language, category) {
    return data.cells.find((cell) => cell.condition === condition && cell.model === model &&
      cell.language === language && cell.category === category);
  }

  function numbers(data) {
    const [first, second] = data.conditions;
    const [base, tuned] = data.models;
    const box = clear($('numbers'));
    const card = (strong, text) => box.appendChild(el('div', { class: 'number' }, el('strong', { text: strong }), el('span', { text })));

    const byLanguage = data.languages.map((code) => cellOf(data, first.id, base.id, code, 'all')).filter((cell) => cell.n);
    byLanguage.sort((a, b) => b.k / b.n - a.k / a.n);
    const worst = byLanguage[0];
    const best = byLanguage[byLanguage.length - 1];
    card(`${worst.k} of ${worst.n}`, `${languageName(worst.language)} prompts get a harmful answer from the ${base.name}, ` +
      `against ${best.k} of ${best.n} in ${languageName(best.language)} (${first.name}).`);

    const summary = data.summary;
    const before = summary[first.id][base.id].harmful;
    const after = summary[first.id][tuned.id].harmful;
    const change = (data.changes[first.id] || {}).safe;
    const scored = before[1] === after[1] ? `${before[1]}` : `${before[1]} and ${after[1]}`;
    let text = `harmful answers of ${scored} scored prompts, ${base.name} then ${tuned.name} ` +
      `(${percent(before[0], before[1])} then ${percent(after[0], after[1])}${change ? ', ' + pValue(change.p) : ''}).`;
    if (second) {
      const b = summary[second.id][base.id].harmful;
      const a = summary[second.id][tuned.id].harmful;
      text += ` With ${second.name}: ${b[0]} of ${b[1]} then ${a[0]} of ${a[1]}.`;
    }
    card(`${before[0]} → ${after[0]}`, text);

    const safeBefore = summary[first.id][base.id].over_refusal;
    const safeAfter = summary[first.id][tuned.id].over_refusal;
    card(`${safeBefore[0]} → ${safeAfter[0]} of ${safeBefore[1]}`,
      'refusals of safe XSTest prompts: over-refusal did not move after the fine-tune.');
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
    td.appendChild(document.createTextNode(`${cell.k} of ${cell.n} (${percent(cell.k, cell.n)}; ` +
      `${(100 * cell.lo).toFixed(1)}% to ${(100 * cell.hi).toFixed(1)}%)`));
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

  function verdictTag(labels) {
    if (!labels) return el('span', { class: 'tag unscored', text: 'no row' });
    return el('span', {
      class: 'tag ' + labels.verdict,
      title: `refusal: ${labels.refusal}, response_harm: ${labels.response_harm}`,
      text: labels.verdict,
    });
  }

  function explorer(data) {
    const [base, tuned] = data.models;
    let page = 0;
    options($('ex-language'), [['all', 'all languages'], ...data.languages.map((code) => [code, languageName(code)])]);
    options($('ex-category'), [['all', 'all categories'], ...data.categories.map((name) => [name, name])]);
    options($('ex-condition'), data.conditions.map((condition) => [condition.id, condition.name]));

    const head = el('tr', {}, el('th', { text: 'Prompt id' }), el('th', { text: 'Language' }), el('th', { text: 'Harm categories' }));
    for (const condition of data.conditions) {
      for (const model of data.models) head.appendChild(el('th', { text: `${model.name}, ${condition.name}` }));
    }
    head.appendChild(el('th', { text: 'Change' }));
    clear($('ex-head')).appendChild(head);

    const verdicts = (row, condition) => {
      const labels = row.labels[condition] || {};
      return [(labels[base.id] || {}).verdict || 'unscored', (labels[tuned.id] || {}).verdict || 'unscored'];
    };
    const tests = {
      all: () => true,
      changed: (before, after) => before !== after,
      fixed: (before, after) => before === 'harmful' && after !== 'harmful' && after !== 'unscored',
      broken: (before, after) => after === 'harmful' && before !== 'harmful' && before !== 'unscored',
      both: (before, after) => before === 'harmful' && after === 'harmful',
      any: (before, after) => before === 'harmful' || after === 'harmful',
    };

    function draw() {
      const language = $('ex-language').value;
      const category = $('ex-category').value;
      const condition = $('ex-condition').value;
      const test = tests[$('ex-change').value];
      const prefix = $('ex-id').value.trim().toLowerCase();
      const rows = data.prompts.filter((row) => (language === 'all' || row.language === language) &&
        (category === 'all' || row.categories.includes(category)) &&
        (!prefix || row.id.startsWith(prefix)) && test(...verdicts(row, condition)));
      const pages = Math.max(1, Math.ceil(rows.length / PAGE));
      page = Math.min(page, pages - 1);
      const body = clear($('ex-body'));
      for (const row of rows.slice(page * PAGE, (page + 1) * PAGE)) {
        const file = `https://huggingface.co/datasets/${data.eval_repo}/blob/main/prompts/harmful/${row.language}.jsonl`;
        const line = el('tr', {},
          el('td', { class: 'id' }, el('a', { href: file, title: row.id, text: row.id.slice(0, 12) })),
          el('td', { text: languageName(row.language) }),
          el('td', { text: row.categories.join('; ') }));
        for (const each of data.conditions) {
          for (const model of data.models) line.appendChild(el('td', {}, verdictTag((row.labels[each.id] || {})[model.id])));
        }
        const [before, after] = verdicts(row, condition);
        line.appendChild(el('td', { text: before === after ? 'same' : `${before} → ${after}` }));
        body.appendChild(line);
      }
      $('ex-count').textContent = `${rows.length} prompts, page ${page + 1} of ${pages}`;
      $('ex-prev').disabled = page === 0;
      $('ex-next').disabled = page >= pages - 1;
    }
    const reset = () => { page = 0; draw(); };
    for (const id of ['ex-language', 'ex-category', 'ex-condition', 'ex-change']) $(id).addEventListener('change', reset);
    $('ex-id').addEventListener('input', reset);
    $('ex-prev').addEventListener('click', () => { page -= 1; draw(); });
    $('ex-next').addEventListener('click', () => { page += 1; draw(); });
    draw();
  }

  fetch('data.json')
    .then((response) => {
      if (!response.ok) throw new Error('HTTP ' + response.status);
      return response.json();
    })
    .then((data) => {
      numbers(data);
      figures(data);
      map(data);
      explorer(data);
      $('status').textContent = `${data.prompts.length} prompts, ${data.models.length} models, ${data.conditions.length} conditions.`;
    })
    .catch((error) => {
      $('status').textContent = 'The data file could not be loaded: ' + error.message;
    });
})();
