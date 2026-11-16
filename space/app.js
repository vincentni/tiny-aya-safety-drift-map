'use strict';

// Reads data.json (counts and intervals only) and fills the page. No dependencies, no network beyond that file.
(function () {
  const LANGUAGE_NAMES = { en: 'English', fr: 'French', es: 'Spanish', ru: 'Russian', hi: 'Hindi', ar: 'Arabic', nl: 'Dutch', tr: 'Turkish' };
  const FIGURE_TITLES = {
    f1: 'Harmful answers by language, base model and fine-tuned model',
    f2: 'Harmful answers of the base model by harm category and language',
    f3: 'Harmful answers, over-refusal and task score, before and after the fine-tune',
    f4: 'Harmful answers by harm category',
    f5: 'How verdicts moved between the base model and the fine-tuned model',
    f6: 'Answers cut off at the token limit, and harmful answers among them',
    f7: 'Harmful answers on native, romanized and code-mixed prompts',
    f8: 'Harmful answers with a targeted system-prompt patch',
    f9: 'Per language on official items, with BBQ\'s English test set apart in the last column: how often a person is ' +
      'named where the text does not say, and twin accuracy, for the base model and the attribution fine-tune, with 95% ' +
      'Wilson intervals. The fine-tune saw English rows only.',
    f10: 'Base model against attribution fine-tune: how often a person is named where the text does not say, and twin ' +
      'accuracy, on official items in four languages (top) and on BBQ\'s English test (bottom), which is not pooled with them.',
  };
  const LEAD_FIGURE = 'f8';
  // the figures of the question pairs, shown with them in this order
  const TOP_FIGURES = ['f10', 'f9'];
  // the pooled scopes of the choice table
  const SCOPE_NAMES = { 'non-English': 'Spanish, Dutch and Turkish', all: 'All four languages' };
  const SMALL = 15;
  const PROBE_NAMES = { squad_probe: 'Probe', length: 'Passage length', tfidf: 'TF-IDF of the prompt',
    probe_1: 'Answerability probe fitted on question pairs', error_probe: 'Error probe fitted on question pairs' };
  const PROBE_TEXT = { length: 'passage length', tfidf: 'TF-IDF of the prompt' };
  const CONFIDENCE_NAMES = { mean_log_probability: 'Mean log-probability of the answer as the gate',
    first_token_probability: 'Probability of the answer\'s first token as the gate' };
  const CONFIDENCE_TEXT = { mean_log_probability: 'the mean log-probability of the answer',
    first_token_probability: 'the probability of its first token' };
  const SAMPLE_NAMES = { first: 'First sample', replication: 'Second sample', pooled: 'Both samples' };

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

  // the question pairs: one count of the choice table, and the change the table states from the base model to an arm
  const choiceRate = (found, arm, scope, metric) => found.rates.find((row) => row.arm === arm && row.scope === scope && row.metric === metric);
  const choiceChange = (found, arm, scope, metric) => found.changes.find((row) => row.arm === arm && row.scope === scope && row.metric === metric);

  function numbers(data) {
    const first = data.conditions[0];
    const [base, tuned] = data.models;
    const found = data.attribution;
    const cards = clear($('numbers'));
    // with the probe on the page its cards stand first, and those of the question pairs under their own heading
    const top = data.probe ? clear($('pair-numbers')) : cards;
    // with the question pairs on the page, the cards of the harmful-request map sit under its own heading
    const box = found ? clear($('map-numbers')) : top;
    const put = (where, strong, text) => where.appendChild(el('div', { class: 'number' }, el('strong', { text: strong }), el('span', { text })));
    const card = (strong, text) => put(box, strong, text);
    if (data.probe) {
      const probed = data.probe;
      const answerable = probed.questions - probed.unanswerable;
      put(cards, `${probed.base.wrong} → ${probed.gated.wrong}`,
        `wrong answers on ${probed.questions} SQuAD 2.0 questions, ${probed.unanswerable} of which their passage cannot answer, ` +
        `when the model abstains where the probe fires` +
        (probed.judge ? ` (${probed.judge.base.wrong} → ${probed.judge.gated.wrong} as a closed judge reads the answers)` : '') +
        `. Abstaining on every question gives ${probed.always_abstain}.`);
      if (probed.instructed) {
        put(cards, `${probed.instructed.counts.wrong} with one instruction line`,
          `in the prompt and no probe: about two thirds of the reduction for free. The probe is ` +
          `${(100 * probed.instructed.behind_the_gate.delta).toFixed(1)} points better than it.`);
      } else {
        put(cards, `${probed.base.hallucinations} → ${probed.gated.hallucinations}`,
          `of the ${probed.unanswerable} unanswerable questions are answered: the base model answers ` +
          `${percent(probed.base.hallucinations, probed.unanswerable)} of the questions it should not.`);
      }
      put(cards, `${probed.gated.kept} of ${answerable}`,
        `answerable questions are still answered (${percent(probed.gated.kept, answerable)}); ${probed.gated.correct} correctly, ` +
        `against the base model's ${probed.base.correct}.`);
      const [own, ...texts] = probed.scores;
      put(cards, `AUC ${own.auc_error.toFixed(2)}`,
        `of the probe against the model's own error; ${texts.map((score) => `${PROBE_TEXT[score.id]} ` +
          `${score.auc_error.toFixed(2)}`).join(', ')}: the text alone does not predict it.`);
    }
    const named = found && ['base', 'tuned'].map((side) => choiceRate(found, found[side], 'all', 'person_pick'));
    if (named && named[0] && named[1]) {
      const both = ['base', 'tuned'].map((side) => choiceRate(found, found[side], 'all', 'twin'));
      const outside = choiceRate(found, found.tuned, 'non-English', 'person_pick');
      put(top, `${percent(named[0].k, named[0].n)} → ${percent(named[1].k, named[1].n)}`,
        `of the who-did-it questions a short text cannot answer get a named person, on official items in four languages: ` +
        `${named[0].k} of ${named[0].n} from the base model, ${named[1].k} after a LoRA fine-tune on English rows` +
        (outside ? `, ${outside.k === named[1].k ? 'all' : outside.k} of them in the three languages it never trained on.` : '.'));
      const apart = found.scopes.filter((scope) => !SCOPE_NAMES[scope])
        .map((scope) => [scope, choiceRate(found, found.tuned, scope, 'twin')]).filter(([, cell]) => cell && cell.n)
        .map(([scope, cell]) => `${percent(cell.k, cell.n)} in ${languageName(scope)}`).join(', ');
      put(top, `${percent(both[0].k, both[0].n)} → ${percent(both[1].k, both[1].n)}`,
        `twin accuracy: the question and its answerable twin both right, ${both[0].k}, then ${both[1].k} of ${both[1].n} pairs, ` +
        `so abstaining on everything scores zero. After the fine-tune: ${apart}.`);
      const test = data.bbq && ['base', 'tuned'].map((side) => ['person_pick', 'twin'].map((metric) => choiceRate(data.bbq, data.bbq[side], 'en', metric)));
      if (test && test.flat().every(Boolean)) {
        put(top, `${test[0][0].k} → ${test[1][0].k} of ${test[1][0].n}`,
          `cases with a named person on BBQ's English test, the benchmark the training rows come from; twin accuracy ` +
          `${percent(test[0][1].k, test[0][1].n)} → ${percent(test[1][1].k, test[1][1].n)}. It stands beside the four-language ` +
          `result and is not pooled with it.`);
      }
      const prompted = found.arms.filter((arm) => !arm.fine_tune && arm.id !== found.base)
        .map((arm) => [arm, choiceRate(found, arm.id, 'all', 'twin')]).filter(([, cell]) => cell && cell.n)
        .sort((a, b) => b[1].k / b[1].n - a[1].k / a[1].n)[0];
      if (prompted) {
        put(top, `Prompts alone: ${percent(prompted[1].k, prompted[1].n)}`,
          `twin accuracy of the best arm that changes the prompt and not the weights (${prompted[0].name}), ` +
          `${prompted[1].k} of ${prompted[1].n} pairs.`);
      }
    }
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
        `harmful answers of the ${base.name} with five sentences of guidance in its system prompt, on the prompts the ` +
        `guidance was written against (${without}, then ` +
        `${withIt} of ${patched.pairs} prompts, ${pValue(patched.p)}). The price: ${refused.refusal[1]} of ` +
        `${refused.pairs} safe prompts refused, against ${refused.refusal[0]}.${beside}`);
    }

    const ranked = (condition) => byRate(data.languages.map((code) => cellOf(data, condition.id, base.id, code, 'all')));
    const languages = ranked(first);
    const worst = languages[0];
    const best = languages[languages.length - 1];
    const second = data.conditions[1];
    const ranking = second ? ranked(second) : [];
    const safest = ranking[ranking.length - 1];
    if (safest && safest.language === best.language) {
      // the safest language holds in both conditions; whether the weakest does is said in a clause
      const weakest = ranking[0].language === worst.language ?
        `${languageName(worst.language)} is the weakest in both` : 'the weakest language changes with the condition';
      card(`${languageName(best.language)}: ${percent(best.k, best.n)}`,
        `harmful answers of the ${base.name} in ${languageName(best.language)} with the ${first.name} (${best.k} of ` +
        `${best.n}), its safest language in both prompt conditions; ${weakest}.`);
    } else {
      card(`${percent(worst.k, worst.n)} against ${percent(best.k, best.n)}`,
        `harmful answers of the ${base.name} with the ${first.name}, in ${languageName(worst.language)} (${worst.k} of ${worst.n}; ${bounds(worst)}) ` +
        `and in ${languageName(best.language)} (${best.k} of ${best.n}; ${bounds(best)}).`);
    }

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
    // the first run is in the condition of the repair
    const [main, ...others] = data.overfit.runs;
    const where = conditionOf(data, main.condition);
    const start = data.summary[where.id][base.id].harmful;
    const end = data.summary[where.id][data.overfit.model].harmful;
    const beside = others.map((run) => {
      const [before, after] = [base.id, data.overfit.model].map((id) => data.summary[run.condition][id].harmful);
      return ` ${capital(conditionOf(data, run.condition).name)}: ${percent(before[0], before[1])} → ` +
        `${percent(after[0], after[1])}.`;
    }).join('');
    const said = (change) => (change ? ` (${pValue(change.p)})` : '');
    if (!data.repaired) {
      card(`${percent(start[0], start[1])} → ${percent(end[0], end[1])}`,
        `harmful answers of the ${base.name} and of the overfit fine-tune, with ${where.name}: the overfit recipe ` +
        `erodes safety${said(main.changes.safe)}.${beside}`);
      return;
    }
    const mended = data.summary[where.id][data.repaired.model].harmful;
    card(`${percent(start[0], start[1])} → ${percent(end[0], end[1])} → ${percent(mended[0], mended[1])}`,
      `harmful answers of the ${base.name}, the overfit fine-tune and the repaired fine-tune, with ${where.name}: ` +
      `overfitting erodes safety${said(main.changes.safe)}, mixing a quarter of the base weights back restores it` +
      `${said(data.repaired.changes.safe)}.${beside}`);
  }

  // the tables state the change in safe rate; harmful answers move the other way
  function stated(change) {
    return change ? `${points(-change.delta)} points, 95% interval ${points(-change.ci[1])} to ` +
      `${points(-change.ci[0])}, ${pValue(change.p)}` : 'no paired change in the tables';
  }

  function recipes(data) {
    if (!data.overfit) return;
    const [main, ...others] = data.overfit.runs;
    const where = conditionOf(data, main.condition);
    const shown = data.models.filter((model) => data.summary[where.id][model.id]);
    const titles = shown.map((model) => el('th', { text: capital(model.name) }));
    fill('recipes-head', [el('tr', {}, el('th', { text: 'Language' }), ...titles)]);
    fill('recipes-body', [...data.languages, 'all'].map((code) => el('tr', {},
      el('td', { text: code === 'all' ? 'all' : languageName(code) }),
      ...shown.map((model) => rateCell(cellOf(data, where.id, model.id, code, 'all'))))));
    const flips = main.flips;
    $('recipes-intro').textContent = `Harmful answers and their 95% Wilson intervals with ${where.name}` +
      (data.repaired ? ', the condition the repair is measured in' : '') + '. Both fine-tunes use the same benign rows: ' +
      'the light recipe runs 3 epochs, the overfit recipe 10 at a higher learning rate.' +
      (data.repaired ? ' The repaired fine-tune is the overfit one with a quarter of each ' +
        'weight taken back from the base model (weight interpolation, no retraining).' : '');
    const cut = (model, condition = where.id) => (data.length[condition] || {})[model];
    const short = (entry, name) => (entry ? ` Of the ${name}'s answers ${counted(...entry.cut_off)} are cut off at the ` +
      `limit, and ${counted(...entry.harmful_complete)} of its complete answers are harmful.` : '');
    let note = `Against the ${data.models[0].name}, ${flips.not_to_harmful} prompts went from a safe answer to a harmful ` +
      `one under the overfit recipe and ${flips.harmful_to_not} the other way (${stated(main.changes.safe)}).` +
      short(cut(data.overfit.model), 'overfit fine-tune');
    if (data.repaired) {
      const mended = data.repaired.flips;
      note += ` After the repair ${mended.harmful_to_not} of the overfit fine-tune's harmful answers are safe and ` +
        `${mended.not_to_harmful} new ones appear (${stated(data.repaired.changes.safe)}).` +
        short(cut(data.repaired.model), 'repaired fine-tune');
    }
    for (const run of others) {
      const [before, after] = [data.models[0].id, data.overfit.model].map((id) => data.summary[run.condition][id].harmful);
      note += ` ${capital(conditionOf(data, run.condition).name)}: the overfit fine-tune gives ${counted(...after)} harmful ` +
        `answers against the ${data.models[0].name}'s ${counted(...before)} (${stated(run.changes.safe)}): ` +
        `${run.flips.not_to_harmful} prompts went from a safe answer to a harmful one and ${run.flips.harmful_to_not} ` +
        'the other way.' + short(cut(data.overfit.model, run.condition), 'overfit fine-tune');
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

    const measures = [['harmful_complete', 'End on their own: harmful'], ['harmful_cut_off', 'Cut off at the limit: harmful'],
      ['cut_off', 'Cut off at the limit, all answers']];
    const titles = ['Model', 'Condition', ...measures.map(([, label]) => label)];
    clear($('length-head')).appendChild(el('tr', {}, ...titles.map((text) => el('th', { text }))));
    const body = clear($('length-body'));
    for (const condition of data.conditions) {
      for (const model of data.models) {
        const entry = (data.length[condition.id] || {})[model.id];
        // a model that was not run in a condition has no row
        if (!entry) continue;
        const counts = measures.map(([key]) => el('td', { class: 'count', text: counted(...entry[key]) }));
        body.appendChild(el('tr', {}, el('td', { text: capital(model.name) }), el('td', { text: condition.name }), ...counts));
      }
    }
    const flips = data.flips[first.id];
    $('length-note').textContent = `Under the ${first.name}, ${flips.harmful_to_not} prompts went from a harmful answer to a safe ` +
      `one after the fine-tune and ${flips.not_to_harmful} went the other way; ${flips.harmful_both} are harmful for both models.`;
    $('length-section').hidden = false;
  }

  function figures(data) {
    // a figure file is named f<number>_<what>
    const key = (file) => file.split('_')[0];
    const shown = (file, lazy) => {
      const title = FIGURE_TITLES[key(file)] || file;
      const image = el('img', { src: 'figures/' + file, alt: title, ...(lazy ? { loading: 'lazy' } : {}) });
      return el('figure', {}, image, el('figcaption', { text: title }));
    };
    const lead = data.figures.find((file) => key(file) === LEAD_FIGURE);
    if (lead) {
      clear($('lead-figure')).appendChild(shown(lead, false));
      $('lead-figure').hidden = false;
    }
    const top = data.attribution ? TOP_FIGURES.map((name) => data.figures.find((file) => key(file) === name)).filter(Boolean) : [];
    if (top.length) {
      const box = clear($('top-figure'));
      for (const file of top) box.appendChild(shown(file, false));
      $('top-figure').hidden = false;
    }
    const rest = data.figures.filter((file) => file !== lead && !TOP_FIGURES.includes(key(file)));
    if (!rest.length) return;
    const box = clear($('figures'));
    for (const file of rest) box.appendChild(shown(file, true));
    $('figures-section').hidden = false;
  }

  function attribution(data) {
    const found = data.attribution;
    if (!found) return;
    const scopeName = (scope) => SCOPE_NAMES[scope] || languageName(scope);
    // inside a sentence the pooled scope is lower case, a list of languages keeps its capitals
    const scopeText = (scope) => (scope === 'all' ? scopeName(scope).toLowerCase() : scopeName(scope));
    const nameOf = (id) => found.arms.find((arm) => arm.id === id).name;
    // the two arms of the headline stand first, side by side
    const pair = [found.base, found.tuned];
    const arms = pair.map((id) => found.arms.find((arm) => arm.id === id)).filter(Boolean)
      .concat(found.arms.filter((arm) => !pair.includes(arm.id)));
    const table = (id, metric, good) => {
      fill(`${id}-head`, [el('tr', {}, el('th', { text: 'Language' }), ...arms.map((arm) => el('th', { text: capital(arm.name) })))]);
      fill(`${id}-body`, found.scopes.map((scope) => el('tr', {}, el('td', { text: scopeName(scope) }),
        ...arms.map((arm) => {
          const cell = choiceRate(found, arm.id, scope, metric);
          const td = rateCell(cell, 1, good);
          // the strict reading stands beside the count wherever it differs
          if (cell && cell.strict !== cell.k) {
            td.appendChild(el('div', { class: 'interval', text: `strict: ${counted(cell.strict, cell.n)}` }));
          }
          return td;
        }))));
    };
    const stated = (scope, metric) => {
      const change = choiceChange(found, found.tuned, scope, metric);
      return change ? `${scopeText(scope)} ${points(change.delta)} points (95% interval ${points(change.ci[0])} to ` +
        `${points(change.ci[1])}, ${pValue(change.p)})` : null;
    };
    const changes = (metric) => ['all', 'non-English'].map((scope) => stated(scope, metric)).filter(Boolean).join('; ');
    table('attr-person', 'person_pick', false);
    table('attr-twin', 'twin', true);
    $('attr-person-note').textContent = 'Counts over the ambiguous items, with 95% Wilson intervals; lower is better. ' +
      'The intervals treat rows as independent; the pooled rows hold the same 150 items in several languages, so theirs ' +
      'are somewhat narrow. ' +
      `Change from the ${nameOf(found.base)} to the ${nameOf(found.tuned)}, paired by item: ${changes('person_pick')}.`;
    const right = ['base', 'tuned'].map((side) => choiceRate(found, found[side], 'all', 'accuracy'));
    // the best arm that changes the prompt and not the weights
    const prompted = found.arms.filter((arm) => !arm.fine_tune && arm.id !== found.base)
      .map((arm) => [arm, choiceRate(found, arm.id, 'all', 'twin')]).filter(([, cell]) => cell && cell.n)
      .sort((a, b) => b[1].k / b[1].n - a[1].k / a[1].n)[0];
    $('attr-twin-note').textContent = 'A pair counts only when the model says the text does not tell on the ambiguous item ' +
      'and answers its twin correctly, so abstaining on everything scores zero; higher is better. ' +
      `Change from the ${nameOf(found.base)} to the ${nameOf(found.tuned)}, paired by item: ${changes('twin')}. ` +
      (right[0] && right[1] ? `Answerable twins answered correctly: ${counted(right[0].k, right[0].n)}, then ${counted(right[1].k, right[1].n)}. ` : '') +
      (prompted ? `The best arm that changes the prompt and not the weights (${prompted[0].name}) reaches ` +
        `${percent(prompted[1].k, prompted[1].n)} over ${scopeText('all')}.` : '');
    englishTest(data.bbq);
    $('attribution-section').hidden = false;
  }

  // a risk-coverage curve as an inline SVG: one point per threshold, the chosen one filled, the base model a square
  function curve(found) {
    const ns = 'http://www.w3.org/2000/svg';
    const svg = (tag, attrs, text) => {
      const node = document.createElementNS(ns, tag);
      for (const [name, value] of Object.entries(attrs)) node.setAttribute(name, value);
      if (text) node.textContent = text;
      return node;
    };
    const [width, height, left, bottom, top, right] = [560, 320, 52, 44, 12, 14];
    const x = (coverage) => left + coverage * (width - left - right);
    const top_risk = 0.6;
    const y = (risk) => height - bottom - (risk / top_risk) * (height - bottom - top);
    const root = svg('svg', { viewBox: `0 0 ${width} ${height}`, role: 'img', 'aria-label': 'Risk among the answered questions against the share of questions answered, by threshold' });
    for (const tick of [0, 0.2, 0.4, 0.6, 0.8, 1]) {
      root.appendChild(svg('line', { x1: x(tick), x2: x(tick), y1: y(0), y2: y(top_risk), stroke: '#e6e9ee' }));
      root.appendChild(svg('text', { x: x(tick), y: height - bottom + 16, 'text-anchor': 'middle', 'font-size': 12, fill: '#5b6472' }, `${100 * tick}%`));
    }
    for (const tick of [0, 0.2, 0.4, 0.6]) {
      root.appendChild(svg('line', { x1: x(0), x2: x(1), y1: y(tick), y2: y(tick), stroke: '#e6e9ee' }));
      root.appendChild(svg('text', { x: left - 8, y: y(tick) + 4, 'text-anchor': 'end', 'font-size': 12, fill: '#5b6472' }, `${Math.round(100 * tick)}%`));
    }
    root.appendChild(svg('text', { x: x(0.5), y: height - 6, 'text-anchor': 'middle', 'font-size': 12, fill: '#1c2330' }, 'Questions answered (coverage)'));
    root.appendChild(svg('text', { x: 14, y: y(top_risk / 2), 'text-anchor': 'middle', 'font-size': 12, fill: '#1c2330', transform: `rotate(-90 14 ${y(top_risk / 2)})` }, 'Wrong among the answered (risk)'));
    const drawn = found.curve.filter((row) => row.risk !== null);
    root.appendChild(svg('polyline', { points: drawn.map((row) => `${x(row.coverage)},${y(row.risk)}`).join(' '), fill: 'none', stroke: '#2a5db0', 'stroke-width': 2 }));
    for (const row of drawn) {
      const chosen = row.threshold === found.threshold;
      root.appendChild(svg('circle', { cx: x(row.coverage), cy: y(row.risk), r: chosen ? 6 : 3, fill: chosen ? '#2a5db0' : '#fff', stroke: '#2a5db0', 'stroke-width': 1.5 }));
    }
    const base = [found.base.answered / found.questions, found.base.wrong_answered / found.base.answered];
    root.appendChild(svg('rect', { x: x(base[0]) - 5, y: y(base[1]) - 5, width: 10, height: 10, fill: '#b3261e' }));
    clear($('probe-curve')).appendChild(root);
    const chosen = found.curve.find((row) => row.threshold === found.threshold);
    $('probe-curve-note').textContent = 'Each point is one threshold of the grid. The filled point is the threshold chosen on the ' +
      `probe's dev split (${found.threshold}): ${(100 * chosen.coverage).toFixed(1)}% of the questions answered, ` +
      `${(100 * chosen.risk).toFixed(1)}% of them wrong. The square is the base model: ${(100 * base[0]).toFixed(1)}% answered, ` +
      `${(100 * base[1]).toFixed(1)}% of them wrong. Half of the questions should not be answered, so coverage near 50% is the bound.`;
  }

  // the probe on SQuAD 2.0: the base model, the gated model and abstaining on every question, side by side
  function probe(data) {
    const found = data.probe;
    if (!found) return;
    const answerable = found.questions - found.unanswerable;
    const always = { wrong: found.always_abstain, answered: 0, wrong_answered: 0, hallucinations: 0, kept: 0, correct: 0 };
    const systems = [['Base model', found.base]];
    if (found.instructed) systems.push(['One instruction line, no probe', found.instructed.counts]);
    systems.push(['Base model + probe', found.gated]);
    if (found.combined) systems.push(['Probe gate, else the instructed answer (exploratory)', found.combined.counts]);
    systems.push(['Abstaining on every question', always]);
    const rows = [
      ['Wrong answers', (counts) => counted(counts.wrong, found.questions)],
      ['Unanswerable questions answered (hallucinations)', (counts) => counted(counts.hallucinations, found.unanswerable)],
      ['Answerable questions still answered', (counts) => counted(counts.kept, answerable)],
      ['Answerable questions answered correctly', (counts) => counted(counts.correct, answerable)],
      ['Questions answered (coverage)', (counts) => counted(counts.answered, found.questions)],
      ['Wrong among the answered (risk)', (counts) => (counts.answered ? counted(counts.wrong_answered, counts.answered) : 'none answered')],
    ];
    const count = (text) => el('td', { class: 'count', text });
    $('probe-lead').textContent = `${found.questions} questions of the official SQuAD 2.0 validation file, each asked with its ` +
      `passage: ${found.unanswerable} that the passage cannot answer and ${answerable} that it can. The probe is a logistic ` +
      `regression on the layer-${found.layer} hidden state at the last prompt token, trained on the model's own errors on questions ` +
      'of the train file. Where it fires the answer becomes an abstention: right on an unanswerable question, wrong on an answerable one.';
    fill('probe-head', [el('tr', {}, el('th', { text: '' }), ...systems.map(([name]) => el('th', { text: name })))]);
    fill('probe-body', rows.map(([title, cell]) => el('tr', {}, el('td', { text: title }), ...systems.map(([, counts]) => count(cell(counts))))));
    const change = found.change;
    $('probe-note').textContent = `Scored by word match. Change in wrong answers ${points(change.delta)} points (95% interval over ` +
      `${change.articles} articles ${points(change.ci[0])} to ${points(change.ci[1])}; ${change.better} of ${change.articles} articles ` +
      `better; ${pValue(change.p)}). The gate fixes ${change.fixed} of the ${found.unanswerable} unanswerable questions and breaks ` +
      `${change.broken} of the ${answerable} answerable ones: at these rates it cuts wrong answers only where more than ` +
      `${(100 * found.break_even).toFixed(1)}% of the questions are unanswerable.`;
    const checks = [];
    if (found.served) {
      checks.push(`Served through a forward hook during generation, the gate gives the offline gate's answers on ` +
        `${found.served.equal} of ${found.served.prompts} prompts (${found.served.fired} fired).`);
    }
    if (found.identity) {
      checks.push(`The ${found.identity.rows} scored answers, generated again through AuditKIT, equal the stored ones on ` +
        `${found.identity.equal} of ${found.identity.rows} rows.`);
    }
    $('probe-checks').textContent = checks.join(' ');
    $('probe-checks').hidden = !checks.length;
    // a table may state a bound with four decimals: shown as stated, not rounded again
    const exact = (value) => (value >= 0 ? '+' : '') + (100 * value).toFixed(2).replace(/0$/, '');
    if (found.instructed) {
      const behind = found.instructed.behind_the_gate;
      $('probe-note').textContent += ` One instruction line in the prompt, with no probe, takes wrong answers to ` +
        `${found.instructed.counts.wrong} (${found.instructed.judge_wrong} as a closed judge reads the answers); the probe is ` +
        `${exact(behind.delta).slice(1)} points better (95% interval over ${behind.articles} articles ${exact(behind.ci[0]).slice(1)} to ` +
        `${exact(behind.ci[1]).slice(1)}; ${pValue(behind.p)}).` +
        (found.combined ? ' The gate with the instructed answer where it does not fire is an exploratory computation, added after the other columns.' : '');
    }
    if (found.samples.length) {
      const first = { id: 'first', questions: found.questions, base: found.base, gated: found.gated, change: found.change,
        judge: found.judge && { base: found.judge.base.wrong, gated: found.judge.gated.wrong } };
      const shown = [first, ...found.samples];
      const title = (sample) => `${SAMPLE_NAMES[sample.id]}, ${sample.questions} questions`;
      const lines = [['Wrong answers', (sample) => `${sample.base.wrong} → ${sample.gated.wrong}`],
        ['Wrong answers as read by the judge', (sample) => (sample.judge ? `${sample.judge.base} → ${sample.judge.gated}` : 'not read')],
        ['Unanswerable questions answered (hallucinations)', (sample) => `${sample.base.hallucinations} → ${sample.gated.hallucinations}`],
        ['Answerable questions still answered', (sample) => `${sample.base.kept} → ${sample.gated.kept}`]];
      fill('probe-samples-head', [el('tr', {}, el('th', { text: 'Base model → base model + probe' }),
        ...shown.map((sample) => el('th', { text: title(sample) })))]);
      fill('probe-samples-body', lines.map(([name, cell]) => el('tr', {}, el('td', { text: name }),
        ...shown.map((sample) => count(cell(sample))))));
      $('probe-samples-note').textContent = 'Change in wrong answers, in points, with its 95% interval over articles: ' +
        shown.map((sample) => `${SAMPLE_NAMES[sample.id].toLowerCase()} ${exact(sample.change.delta)} (${exact(sample.change.ci[0])} to ` +
          `${exact(sample.change.ci[1])})`).join('; ') + '.';
      $('probe-samples').hidden = false;
    }
    if (found.judge) {
      const judged = found.judge;
      const metrics = [['Wrong answers', 'wrong'], ['Unanswerable questions answered (hallucinations)', 'hallucinations'],
        ['Answerable questions still answered', 'kept'], ['Answerable questions answered correctly', 'correct']];
      const pair = (base, gated, key) => `${base[key]} → ${gated[key]}`;
      fill('probe-judge-head', [el('tr', {}, el('th', { text: 'Base model → base model + probe' }),
        el('th', { text: 'By word match' }), el('th', { text: 'As read by the judge' }))]);
      fill('probe-judge-body', metrics.map(([title, key]) => el('tr', {}, el('td', { text: title }),
        count(pair(found.base, found.gated, key)), count(pair(judged.base, judged.gated, key)))));
      const stated = judged.change;
      $('probe-judge-note').textContent = `Under the judge's labels the change in wrong answers is ${exact(stated.delta)} points ` +
        `(95% interval over ${stated.articles} articles ${exact(stated.ci[0])} to ${exact(stated.ci[1])}; ${stated.better} of ` +
        `${stated.articles} articles better), and the gate breaks ${stated.broken} answerable questions against ${change.broken} by ` +
        `word match. The two readings agree on ${judged.agree.wrong[0]} of ${judged.agree.wrong[1]} error labels, on ` +
        `${judged.agree.abstains[0]} of ${judged.agree.abstains[1]} abstentions and, on the answerable questions, on ` +
        `${judged.agree.gold[0]} of ${judged.agree.gold[1]} answers.`;
      $('probe-judge').hidden = false;
    }
    // one system per row: the instruction line and a fine-tune on the first sample, the model's own confidence on the second
    const beside = [];
    const notes = [];
    const system = (name, sample, counts, judged, auc) => beside.push([name, sample, `${counts.wrong}`,
      judged === null ? 'not read' : `${judged}`, `${counts.hallucinations}`, `${counts.kept}`, auc === null ? '' : auc.toFixed(4)]);
    if (found.instructed || found.fine_tuned) {
      system('Base model', 'first', found.base, found.judge ? found.judge.base.wrong : null, null);
      system('Base model + probe', 'first', found.gated, found.judge ? found.judge.gated.wrong : null, found.scores[0].auc_error);
      if (found.instructed) system('One instruction line in the prompt, no probe', 'first', found.instructed.counts, found.instructed.judge_wrong, null);
      if (found.fine_tuned) {
        const tuned = found.fine_tuned;
        system('Fine-tune on question pairs, no probe', 'first', tuned.counts, tuned.judge_wrong, null);
        const stated = (change) => `${exact(change.delta)} points (95% interval ${exact(change.ci[0])} to ${exact(change.ci[1])})`;
        notes.push(`The fine-tune's change in wrong answers from the base model is ${stated(tuned.from_base.word_match)} by word match and ` +
          `${stated(tuned.from_base.judge)} as the judge reads the answers` +
          (tuned.from_base.judge.ci[0] < 0 && tuned.from_base.judge.ci[1] > 0 ? ', an interval that includes no change.' : '.'));
      }
    }
    if (found.confidence) {
      const sure = found.confidence;
      system('Base model', 'second', sure.base.counts, sure.base.judge_wrong, null);
      system('Base model + probe', 'second', sure.gated.counts, sure.gated.judge_wrong, sure.gated.auc_error);
      for (const score of sure.scores) system(CONFIDENCE_NAMES[score.id], 'second', score.counts, score.judge_wrong, score.auc_error);
      notes.push('On the second sample the probe\'s AUC against the model\'s error is ' + sure.scores.map((score) =>
        `${score.probe_ahead.difference.toFixed(4)} above ${CONFIDENCE_TEXT[score.id]} (95% interval ` +
        `${score.probe_ahead.ci[0].toFixed(4)} to ${score.probe_ahead.ci[1].toFixed(4)})`).join(' and ') + '.');
    }
    if (beside.length) {
      const titles = ['System', 'Sample', 'Wrong answers', 'Wrong answers as read by the judge', 'Unanswerable questions answered',
        'Answerable questions still answered', 'AUC against the model\'s error'];
      fill('probe-beside-head', [el('tr', {}, ...titles.map((text) => el('th', { text })))]);
      fill('probe-beside-body', beside.map(([name, sample, ...cells]) => el('tr', {}, el('td', { text: name }), el('td', { text: sample }),
        ...cells.map((text) => count(text)))));
      $('probe-beside-note').textContent = [`Each sample holds ${found.questions} questions, ${found.unanswerable} of them unanswerable.`,
        ...notes, 'An instruction line and a fine-tune are not scores, so they have no AUC.'].join(' ');
      $('probe-beside').hidden = false;
    }
    const gateRow = (name, score) => el('tr', {}, el('td', { text: name }), count(score.auc_error.toFixed(4)),
      count(score.auc_unanswerable.toFixed(4)), count(`${score.fired[0]} of ${score.fired[1]}`),
      count(`${found.base.wrong} → ${score.wrong}`), count(`${score.kept} of ${answerable}`));
    const columns = ['AUC against the model\'s error', 'AUC against "unanswerable"', 'Fires on', 'Wrong answers', 'Answerable still answered'];
    fill('probe-scores-head', [el('tr', {}, el('th', { text: 'Score' }), ...columns.map((text) => el('th', { text })))]);
    fill('probe-scores-body', found.scores.map((score) => gateRow(score.id === found.scores[0].id ? `Probe on layer ${found.layer}` : PROBE_NAMES[score.id], score)));
    $('probe-scores-note').textContent = found.ahead.map((gap) => `The probe's AUC against the model's error is ` +
      `${gap.difference.toFixed(4)} above ${PROBE_TEXT[gap.of]} (95% interval ${gap.ci[0].toFixed(4)} to ${gap.ci[1].toFixed(4)}).`).join(' ') +
      ' A text score that fires on every question abstains on everything. The probe\'s AUC against "unanswerable" is above its AUC ' +
      'against the model\'s error: trained on an error label, it mostly learns which questions are unanswerable.';
    fill('probe-elsewhere-head', [el('tr', {}, el('th', { text: 'Probe' }), ...columns.map((text) => el('th', { text })))]);
    fill('probe-elsewhere-body', [gateRow('Fitted on SQuAD (the probe above)', found.scores[0]),
      ...found.elsewhere.map((score) => gateRow(PROBE_NAMES[score.id], score))]);
    curve(found);
    $('probe-section').hidden = false;
  }

  // BBQ's English test beside the official items: one row per arm, never pooled with them
  function englishTest(found) {
    if (!found) return;
    const columns = [['person_pick', 'Names a person where the text does not say', false],
      ['accuracy', 'Answerable twins answered correctly', true], ['twin', 'Twin accuracy', true]];
    const pair = [found.base, found.tuned];
    const arms = pair.map((id) => found.arms.find((arm) => arm.id === id)).filter(Boolean)
      .concat(found.arms.filter((arm) => !pair.includes(arm.id)));
    fill('attr-bbq-head', [el('tr', {}, el('th', { text: 'Arm' }), ...columns.map(([, title]) => el('th', { text: title })))]);
    fill('attr-bbq-body', arms.map((arm) => el('tr', {}, el('td', { text: capital(arm.name) }),
      ...columns.map(([metric, , good]) => rateCell(choiceRate(found, arm.id, 'en', metric), 1, good)))));
    const stated = columns.map(([metric, title]) => [title, choiceChange(found, found.tuned, 'en', metric)]).filter(([, change]) => change)
      .map(([title, change]) => `${title.toLowerCase()} ${points(change.delta)} points (95% interval ${points(change.ci[0])} to ` +
        `${points(change.ci[1])}, ${pValue(change.p)})`).join('; ');
    $('attr-bbq-note').textContent = 'Counts over 150 question pairs, with 95% Wilson intervals. Change from the base model to ' +
      `the attribution fine-tune, paired by item: ${stated}.`;
    $('attr-bbq').hidden = false;
  }

  // the sample question pair: the ambiguous question with both answers, and the twin with the fine-tune's answer
  function samples(data, shown) {
    if (!data.attribution || !shown || !shown.samples || !shown.samples.length) return;
    const found = data.attribution;
    const nameOf = (id) => capital(found.arms.find((arm) => arm.id === id).name);
    const answer = (who, badge, label, text) => el('p', {}, el('span', { class: `badge ${badge}`, text: label }),
      el('span', { class: 'tag', text: who }), el('blockquote', { text }));
    const box = clear($('attr-sample-cards'));
    for (const sample of shown.samples) {
      box.appendChild(el('div', { class: 'system', lang: sample.language },
        el('p', {}, el('strong', { text: languageName(sample.language) })),
        el('p', { class: 'tag', text: 'The text does not say:' }), el('blockquote', { text: sample.prompt }),
        answer(nameOf(found.base), 'named', 'names a person', sample.base),
        answer(nameOf(found.tuned), 'abstains', 'says the text does not tell', sample.tuned),
        el('p', { class: 'tag', text: 'The twin, whose text does say:' }), el('blockquote', { text: sample.twin.prompt }),
        answer(nameOf(found.tuned), 'answered', 'answers', sample.twin.tuned)));
    }
    $('attr-samples').hidden = false;
  }

  // a table wider than its box scrolls sideways; a line above it says so
  function hints() {
    for (const box of document.querySelectorAll('.scroll')) {
      let hint = box.previousElementSibling;
      if (!hint || hint.className !== 'hint') {
        hint = el('p', { class: 'hint', text: 'Wider than the screen: scroll sideways →' });
        box.parentNode.insertBefore(hint, box);
      }
      hint.hidden = box.scrollWidth <= box.clientWidth + 1;
    }
  }

  // ``full`` is the share at which the bar is at its longest; a ``good`` bar is one where more is better
  function rateCell(cell, full = 0.4, good = false) {
    const td = el('td', { class: 'count' });
    if (!cell || !cell.n) {
      td.textContent = 'no scored prompts';
      return td;
    }
    const bar = el('span', { class: good ? 'bar good' : 'bar' });
    bar.style.width = Math.round(Math.min(1, cell.k / cell.n / full) * 60) + 'px';
    // the count on one line, the bar and the interval below it, so four models fit side by side
    td.appendChild(el('div', { text: counted(cell.k, cell.n) }));
    td.appendChild(el('div', { class: 'interval' }, bar, bounds(cell)));
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
      hints();
    }
    for (const id of ['map-condition', 'map-language', 'map-category', 'map-small']) $(id).addEventListener('change', draw);
    draw();
  }

  function render(data) {
    numbers(data);
    probe(data);
    attribution(data);
    length(data);
    recipes(data);
    patch(data);
    variants(data);
    figures(data);
    map(data);
    window.addEventListener('resize', hints);
    $('status').textContent = `${data.cells.length} cells, ${data.models.length} models, ${data.conditions.length} conditions.`;
  }

  // without its data the page says how to see it, and shows no half-filled section
  function unavailable(reason) {
    const live = 'https://huggingface.co/spaces/vincentnkq/tiny-aya-safety-map';
    const why = reason ? `The data file could not be loaded (${reason}).` :
      'This page loads its data over HTTP, so it does not work when opened as a file.';
    const parts = [el('strong', { text: why }), ' See it live at ', el('a', { href: live, text: live }),
      ', or serve this folder from the repository root: ', el('code', { text: 'python space/build_data.py' }), ', then ',
      el('code', { text: 'python -m http.server --directory space' }), ', and open ',
      el('code', { text: 'http://localhost:8000/' }), '.'];
    const box = clear($('unavailable'));
    for (const part of parts) box.appendChild(typeof part === 'string' ? document.createTextNode(part) : part);
    box.hidden = false;
    $('content').hidden = true;
    $('status').textContent = '';
  }

  // a browser does not let a page opened as a file fetch another file
  if (location.protocol === 'file:') {
    unavailable();
    return;
  }
  fetch('data.json')
    .then((response) => {
      if (!response.ok) throw new Error('HTTP ' + response.status);
      return response.json();
    })
    .then((data) => {
      render(data);
      // the sample item is text, so it is not in the data file; without it the section shows its tables only
      if (data.attribution) {
        fetch('samples.json').then((response) => (response.ok ? response.json() : null))
          .then((shown) => samples(data, shown)).then(hints).catch(() => {});
      }
    }, (error) => unavailable(error.message))
    .catch((error) => {
      $('status').textContent = 'The page could not be drawn: ' + error.message;
    });
})();
