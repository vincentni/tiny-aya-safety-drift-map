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

  function fill(id, rows) {
    const box = clear($(id));
    for (const row of rows) box.appendChild(row);
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
  const points = (value) => (value >= 0 ? '+' : '-') + Math.abs(100 * value).toFixed(1);
  const pValue = (p) => (p < 0.001 ? 'p < 0.001' : 'p = ' + p.toFixed(3));
  const counted = (k, n) => `${k} of ${n} (${percent(k, n)})`;
  const modelOf = (data, id) => data.models.find((model) => model.id === id);
  const capital = (text) => text[0].toUpperCase() + text.slice(1);
  const conditionOf = (data, id) => data.conditions.find((condition) => condition.id === id);

  // shared with explorer.js
  window.pageKit = { el, clear, options, languageName };

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

    const patched = data.patch.find((entry) => entry.model === base.id);
    if (patched) {
      const [without, withIt] = patched.harmful;
      const refused = patched.over_refusal;
      const other = data.patch.find((entry) => entry.model === tuned.id);
      const beside = other ? ` For the ${tuned.name}: ${other.harmful[0]}, then ${other.harmful[1]} of ` +
        `${other.pairs} (${pValue(other.p)}).` : '';
      card(`${percent(without, patched.pairs)} → ${percent(withIt, patched.pairs)}`,
        `harmful answers of the ${base.name} with five sentences of guidance in its system prompt (${without}, then ` +
        `${withIt} of ${patched.pairs} prompts, ${pValue(patched.p)}). The price: ${refused.refusal[1]} of ` +
        `${refused.pairs} safe prompts refused, against ${refused.refusal[0]}.${beside}`);
    }

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
    card('Light fine-tune: no safety cost',
      `harmful answers of the ${tuned.name} against the ${base.name}: ${after.harmful[0]} of ${after.harmful[1]} against ` +
      `${before.harmful[0]} of ${before.harmful[1]}, with over-refusal unchanged (${after.over_refusal[0]} against ` +
      `${before.over_refusal[0]} of ${before.over_refusal[1]} safe prompts refused). Among complete answers it is not safer.`);

    if (!data.overfit) return;
    const where = conditionOf(data, data.overfit.condition);
    const start = data.summary[where.id][base.id].harmful;
    const end = data.summary[where.id][data.overfit.model].harmful;
    // the tables state the change in safe rate; harmful answers move the other way
    const stated = (change) => (change ? `${points(-change.delta)} points, 95% interval ${points(-change.ci[1])} to ` +
      `${points(-change.ci[0])}, ${pValue(change.p)}` : 'no paired change in the tables');
    if (!data.repaired) {
      card(`${percent(start[0], start[1])} → ${percent(end[0], end[1])}`,
        `harmful answers after the overfit recipe on the same rows, with ${where.name} (${start[0]} of ${start[1]}, then ` +
        `${end[0]} of ${end[1]}; ${stated(data.overfit.changes.safe)}). The overfit recipe erodes safety.`);
      return;
    }
    const mended = data.summary[where.id][data.repaired.model].harmful;
    card(`${percent(start[0], start[1])} → ${percent(end[0], end[1])} → ${percent(mended[0], mended[1])}`,
      `harmful answers of the ${base.name}, the overfit fine-tune and the repaired fine-tune, with ${where.name} ` +
      `(${start[0]} of ${start[1]}, ${end[0]} of ${end[1]}, ${mended[0]} of ${mended[1]}). Overfitting erodes safety ` +
      `(${stated(data.overfit.changes.safe)}); mixing a quarter of the base weights back restores it ` +
      `(${stated(data.repaired.changes.safe)}).`);
  }

  function recipes(data) {
    if (!data.overfit) return;
    const where = conditionOf(data, data.overfit.condition);
    const shown = data.models.filter((model) => data.summary[where.id][model.id]);
    const titles = shown.map((model) => el('th', { text: capital(model.name) }));
    fill('recipes-head', [el('tr', {}, el('th', { text: 'Language' }), ...titles)]);
    fill('recipes-body', [...data.languages, 'all'].map((code) => el('tr', {},
      el('td', { text: code === 'all' ? 'all' : languageName(code) }),
      ...shown.map((model) => rateCell(cellOf(data, where.id, model.id, code, 'all'))))));
    const flips = data.overfit.flips;
    $('recipes-intro').textContent = `Harmful answers with ${where.name}, the one condition the overfit fine-tune is ` +
      'measured in. Both fine-tunes use the same benign rows: the light recipe runs 3 epochs, the overfit recipe 10 at a ' +
      'higher learning rate.' + (data.repaired ? ' The repaired fine-tune is the overfit one with a quarter of each ' +
        'weight taken back from the base model (weight interpolation, no retraining).' : '');
    const cut = (model) => (data.length[where.id] || {})[model];
    const short = (entry, name) => (entry ? ` Of the ${name}'s answers ${counted(...entry.cut_off)} are cut off at the ` +
      `limit, and ${counted(...entry.harmful_complete)} of its complete answers are harmful.` : '');
    let note = `Against the ${data.models[0].name}, ${flips.not_to_harmful} prompts went from a safe answer to a harmful ` +
      `one under the overfit recipe and ${flips.harmful_to_not} the other way.` + short(cut(data.overfit.model), 'overfit fine-tune');
    if (data.repaired) {
      const mended = data.repaired.flips;
      note += ` After the repair ${mended.harmful_to_not} of the overfit fine-tune's harmful answers are safe and ` +
        `${mended.not_to_harmful} new ones appear.` + short(cut(data.repaired.model), 'repaired fine-tune');
    }
    $('recipes-note').textContent = note;
    $('recipes-section').hidden = false;
  }

  function patch(data) {
    if (!data.patch.length) return;
    const head = el('tr', {}, el('th', { text: 'Metric' }));
    // the two kinds of harm the patch is written for
    const targeted = data.categories.filter((name) => /Representation|Self-Harm/.test(name));
    const rows = [
      ['Harmful answers', (entry) => [entry.pairs, entry.harmful]],
      ...targeted.map((name) => [`Harmful answers, ${shortName(name)}`,
        (entry) => [entry.categories[name].pairs, entry.categories[name].harmful]]),
      ['Refusals of harmful prompts', (entry) => [entry.pairs, entry.refusal]],
      ['Refusals of safe prompts (over-refusal)', (entry) => [entry.over_refusal.pairs, entry.over_refusal.refusal]],
    ].map(([label, read]) => [read, el('tr', {}, el('td', { text: label }))]);
    // the task score is a mean over the benign prompts of each condition, not a count over pairs
    const task = el('tr', {}, el('td', { text: 'Task score (chrF)' }));
    const score = ([mean, n]) => (n ? (100 * mean).toFixed(1) : 'n/a');
    const notes = [];
    for (const entry of data.patch) {
      const model = modelOf(data, entry.model);
      head.appendChild(el('th', { text: capital(model.name) }));
      head.appendChild(el('th', { text: `${capital(model.name)} + patch` }));
      for (const [read, row] of rows) {
        const [pairs, sides] = read(entry);
        for (const k of sides) row.appendChild(el('td', { class: 'count', text: counted(k, pairs) }));
      }
      for (const condition of [data.conditions[0].id, 'patched']) {
        task.appendChild(el('td', { class: 'count', text: score(data.summary[condition][model.id].chrf) }));
      }
      const refused = entry.over_refusal.refusal;
      notes.push(`${capital(model.name)}: ${entry.harmful_to_not} harmful answers became safe with the patch and ` +
        `${entry.not_to_harmful} appeared (exact sign test, ${pValue(entry.p)}); it refuses ${refused[1]} of ` +
        `${entry.over_refusal.pairs} safe prompts with the patch against ${refused[0]} without.`);
    }
    fill('patch-head', [head]);
    fill('patch-body', [...rows.map(([, row]) => row), task]);
    $('patch-note').textContent = notes.join(' ');
    $('patch-section').hidden = false;
  }

  function variants(data) {
    if (!data.variants.length) return;
    const arrow = (pair) => `${pair[0]} → ${pair[1]}`;
    const kinds = { romanized: 'in Latin script', codemixed: 'mixed with English' };
    fill('variants-body', data.variants.map((row) => el('tr', {},
      el('td', { text: `${languageName(row.parent_language)} ${kinds[row.kind] || row.kind}` }),
      el('td', { text: modelOf(data, row.model).name }),
      el('td', { class: 'count', text: String(row.pairs) }),
      el('td', { class: 'count', text: arrow(row.refusal) }),
      el('td', { class: 'count', text: arrow(row.harmful) }),
      el('td', { class: 'count', text: arrow(row.answered) }))));
    const left = data.variants.reduce((sum, row) => sum + row.left_out, 0);
    $('variants-note').textContent = `${left} variants the judge read as another request than their parent are left out.`;
    $('variants-section').hidden = false;
  }

  function length(data) {
    const first = data.conditions[0];
    const [base, tuned] = data.models;
    if (!data.length[first.id]) return;
    const cut = (model) => data.length[first.id][model.id].cut_off;
    const complete = (model) => data.length[first.id][model.id].harmful_complete;
    $('mechanism').textContent = 'The light recipe costs no safety, mostly because its answers are shorter (answers cut at the ' +
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
    const base = data.models[0];
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
      // the models measured in this condition; the base model leads and sorts the rows
      const columns = data.models.filter((model) => data.summary[condition][model.id]);
      fill('map-head', [el('tr', {}, el('th', { text: 'Language' }), el('th', { text: 'Harm category' }),
        ...columns.map((model) => el('th', { text: capital(model.name) })))]);
      fill('map-body', shown.map((cell) => el('tr', {},
        el('td', { text: cell.language === 'all' ? 'all' : languageName(cell.language) }),
        el('td', { text: cell.category === 'all' ? 'all' : cell.category }),
        ...columns.map((model) => rateCell(cellOf(data, condition, model.id, cell.language, cell.category))))));
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
      recipes(data);
      patch(data);
      variants(data);
      figures(data);
      map(data);
      $('status').textContent = `${data.cells.length} cells, ${data.models.length} models, ${data.conditions.length} conditions.`;
    })
    .catch((error) => {
      $('status').textContent = 'The data file could not be loaded: ' + error.message;
    });
})();
