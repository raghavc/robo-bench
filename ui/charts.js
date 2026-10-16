/* Results charts: score bars with CIs, run-outcome stacks, Pareto frontier, summary table.
   Plain script (no modules) so the page also works from file://. Exposes window.Charts. */
(function () {
  // Run outcomes are grouped into a few headline buckets; the tooltip breaks "Incorrect" down.
  const GROUPS = [
    { key: 'correct',      label: 'Correct',       color: '#4f8a63', soft: '#9bbca3', of: ['success'] },
    { key: 'incorrect',    label: 'Incorrect',     color: '#b4554b', soft: '#d6a39c', hatch: true,
      of: ['decoy', 'bystander_bag', 'not_a_bag', 'wrong_location', 'right_room_wrong_view'] },
    { key: 'gave_up',      label: 'Gave up',       color: '#7a7672', soft: '#b3b0ac', of: ['gave_up'] },
    { key: 'budget',       label: 'Out of budget', color: '#d38b3c', soft: '#e7c19a', of: ['budget_exhausted'] },
    { key: 'turns',        label: 'Turn limit',    color: '#8e7cc3', soft: '#c3b9e0', of: ['turn_limit'] },
    { key: 'disqualified', label: 'Disqualified',  color: '#2f2c29', soft: '#8d8984', hatch: true, of: ['disqualified'] },
  ];
  // Any outcome not named above still gets a segment, so every bar sums to 100%.
  const KNOWN = new Set(GROUPS.flatMap((g) => g.of));
  GROUPS.push({ key: 'other', label: 'Other', color: '#b9b4ad', soft: '#dcd8d2', other: true, of: [] });
  const SUB_LABEL = {
    decoy: 'look-alike decoy', bystander_bag: "someone else's bag", not_a_bag: 'not a bag',
    wrong_location: 'wrong place', right_room_wrong_view: 'right room, wrong view',
  };
  const SCRIPTED = { random: '#c9c4bc', exhaustive: '#a8a29a', clue_reasoner: '#77716a' };

  const pct = (v) => `${(v * 100).toFixed(1)}%`;
  const esc = (s) => String(s ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
  const colorFor = (r) => r.color || SCRIPTED[r.agent] || '#8a847b';
  const agentColor = (a) => a.color || SCRIPTED[a.id] || '#8a847b';
  const sub = (r) => (r.kind === 'scripted' ? 'scripted reference' : r.kind === 'mock' ? 'mock pipeline test'
    : r.harness || '') + (r.n != null ? ` · n=${r.n}` : '');
  const nameCell = (r) => `<span class="bar-label"><b>${esc(r.label)}</b><small>${esc(sub(r))}</small></span>`;
  const gridLines = () => `<div class="bars-grid" aria-hidden="true">${[0, 25, 50, 75, 100].map((p) => `<span style="left:${p}%"></span>`).join('')}</div>`;

  function tooltip() {
    let t = document.getElementById('chart-tip');
    if (!t) {
      t = document.createElement('div');
      t.id = 'chart-tip';
      t.className = 'chart-tip';
      t.setAttribute('role', 'tooltip');
      document.body.appendChild(t);
    }
    return t;
  }
  function bindTip(el, html) {
    const t = tooltip();
    const place = (e) => {
      const x = Math.min(e.clientX + 16, window.innerWidth - t.offsetWidth - 12);
      const y = e.clientY - t.offsetHeight - 14 > 8 ? e.clientY - t.offsetHeight - 14 : e.clientY + 18;
      t.style.left = `${Math.max(8, x)}px`; t.style.top = `${y}px`;
    };
    el.addEventListener('pointerenter', (e) => { t.innerHTML = html(); t.classList.add('show'); place(e); });
    el.addEventListener('pointermove', place);
    el.addEventListener('pointerleave', () => t.classList.remove('show'));
  }
  const tipRow = (color, text, hatch) => `<div class="tip-row"><i class="sq${hatch ? ' hatch' : ''}" style="background-color:${color}"></i>${text}</div>`;

  /* Overall score: thin rounded bars on a fixed 0–100% scale, 95% CI whiskers, name + harness on the right. */
  function bars(el, rows, { value = 'success', ci = 'success_ci', fmt = pct, min = 0, max = 1, onClick } = {}) {
    const span = max - min;
    const pos = (v) => `${Math.max(0, Math.min(1, (v - min) / span)) * 100}%`;
    const base = min < 0 ? 0 : min;
    el.innerHTML = `<div class="lb">${rows.map((r, i) => {
      const c = colorFor(r), v = r[value];
      const lo = Math.min(v, base), hi = Math.max(v, base);
      const w = r[ci] && r[ci][1] - r[ci][0] > 0.004
        ? `<span class="whisker" style="left:${pos(r[ci][0])};width:calc(${pos(r[ci][1])} - ${pos(r[ci][0])})"></span>` : '';
      return `<div class="lb-row${onClick ? ' clickable' : ''}" data-i="${i}">
        <span class="lb-val" style="color:${c}">${fmt(v)}</span>
        <span class="lb-track">${min < 0 ? `<span class="zero" style="left:${pos(0)}"></span>` : ''}
          <span class="lb-fill" style="left:${pos(lo)};width:calc(${pos(hi)} - ${pos(lo)});background:${c}"></span>${w}</span>
        ${nameCell(r)}</div>`;
    }).join('')}<div class="lb-grid">${gridLines()}</div></div>`;
    el.querySelectorAll('.lb-row').forEach((row) => {
      const r = rows[+row.dataset.i];
      bindTip(row, () => `<div class="tip-title">${tipRow(colorFor(r), `<b>${esc(r.label)}</b>`)}</div>
        <div class="tip-row">${value === 'success' ? 'Success' : 'Reward'}: ${fmt(r[value])}</div>
        ${r[ci] ? `<div class="tip-row muted">95% CI: ${fmt(r[ci][0])} – ${fmt(r[ci][1])}</div>` : ''}
        <div class="tip-row muted">Attempts: ${r.n}</div>`);
      if (onClick) row.addEventListener('click', () => onClick(r));
    });
  }

  /* Run outcome: 100% stacked bars; hovering a row fades the others. */
  function stacked(el, rows, _order, { onClick } = {}) {
    const counts = (r, g) => g.other
      ? Object.entries(r.outcomes).reduce((s, [o, n]) => s + (KNOWN.has(o) ? 0 : n), 0)
      : g.of.reduce((s, o) => s + (r.outcomes[o] || 0), 0);
    const used = GROUPS.filter((g) => g.key !== 'other' && g.key !== 'disqualified' || rows.some((r) => counts(r, g)));
    el.innerHTML = `<div class="lb outcome">${rows.map((r, i) => {
      const segs = GROUPS.map((g) => {
        const n = counts(r, g);
        if (!n) return '';
        const share = n / r.n;
        return `<span class="seg${g.hatch ? ' hatch' : ''}" style="flex:${share};--c:${g.color};--s:${g.soft}">${share >= 0.06 ? Math.round(share * 100) : ''}</span>`;
      }).join('');
      return `<div class="lb-row${onClick ? ' clickable' : ''}" data-i="${i}"><span class="lb-track stack">${segs}</span>${nameCell(r)}</div>`;
    }).join('')}<div class="lb-grid">${gridLines()}</div></div>
    <div class="legend">${used.map((g) => `<span><i class="sq${g.hatch ? ' hatch' : ''}" style="background-color:${g.color}"></i>${g.label}</span>`).join('')}</div>`;
    el.querySelectorAll('.lb-row').forEach((row) => {
      const r = rows[+row.dataset.i];
      bindTip(row, () => {
        let html = `<div class="tip-title">${tipRow(GROUPS[0].color, `<b>${esc(r.label)}</b>`)}</div>`;
        for (const g of GROUPS) {
          const n = counts(r, g);
          if (!n && (g.key === 'disqualified' || g.key === 'other')) continue;
          html += tipRow(g.color, `${g.label}: ${n} (${Math.round((n / r.n) * 100)}%)`, g.hatch);
          if (g.key === 'incorrect' && n) {
            html += g.of.filter((o) => r.outcomes[o]).map((o) => `<div class="tip-row sub">${SUB_LABEL[o]}: ${r.outcomes[o]}</div>`).join('');
          }
        }
        return html + `<div class="tip-row">Attempts: ${r.n}</div>`;
      });
      if (onClick) row.addEventListener('click', () => onClick(r));
    });
  }

  /* Pass rate vs cost / tokens / turns, with the Pareto frontier. */
  function pareto(el, rows, { x = 'cost_per_run', y = 'success', onClick } = {}) {
    rows = rows.filter((r) => r[x] != null);
    if (!rows.length) { el.innerHTML = '<div class="empty">No runs with this measure.</div>'; return; }
    const W = 760, H = 420, m = { l: 56, r: 24, t: 20, b: 48 };
    const xs = rows.map((r) => r[x] || 0);
    const xMax = Math.max(...xs, x === 'cost_per_run' ? 0.01 : 1) * 1.15;
    const sx = (v) => m.l + (v / xMax) * (W - m.l - m.r);
    const sy = (v) => H - m.b - v * (H - m.t - m.b);
    const fmtX = x === 'cost_per_run' ? (v) => `$${v.toFixed(v < 0.1 ? 3 : 2)}` : x === 'tokens' ? (v) => v >= 1000 ? `${(v / 1000).toFixed(0)}k` : v.toFixed(0) : (v) => v.toFixed(0);
    const xLabel = { cost_per_run: 'Avg cost (USD / run)', tokens: 'Avg tokens / run', turns: 'Avg turns / run' }[x];
    const sorted = rows.map((r) => ({ r, x: r[x] || 0, y: r[y] })).sort((a, b) => a.x - b.x || b.y - a.y);
    const front = [];
    for (const p of sorted) if (p.y > 0 && p.r.kind !== 'scripted' && (!front.length || p.y > front[front.length - 1].y)) front.push(p);
    let svg = `<svg viewBox="0 0 ${W} ${H}" class="pareto" role="img" aria-label="Pass rate versus ${esc(xLabel)}">`;
    for (let i = 0; i <= 4; i++) {
      const yy = sy(i / 4);
      svg += `<line x1="${m.l}" x2="${W - m.r}" y1="${yy}" y2="${yy}" class="gridline"/><text x="${m.l - 8}" y="${yy + 4}" class="axis" text-anchor="end">${i * 25}%</text>`;
    }
    for (let i = 0; i <= 4; i++) {
      const v = (xMax / 1.15) * i / 4, xx = sx(v);
      svg += `<line x1="${xx}" x2="${xx}" y1="${m.t}" y2="${H - m.b}" class="gridline"/><text x="${xx}" y="${H - m.b + 18}" class="axis" text-anchor="middle">${fmtX(v)}</text>`;
    }
    svg += `<line x1="${m.l}" x2="${m.l}" y1="${m.t}" y2="${H - m.b}" class="axisline"/><line x1="${m.l}" x2="${W - m.r}" y1="${H - m.b}" y2="${H - m.b}" class="axisline"/>`;
    svg += `<text x="${(W + m.l) / 2}" y="${H - 8}" class="axis-title" text-anchor="middle">${esc(xLabel)}</text>`;
    svg += `<text x="14" y="${(H - m.b + m.t) / 2}" class="axis-title" text-anchor="middle" transform="rotate(-90 14 ${(H - m.b + m.t) / 2})">Pass rate</text>`;
    if (front.length > 1) svg += `<polyline class="frontier" points="${front.map((p) => `${sx(p.x)},${sy(p.y)}`).join(' ')} ${W - m.r},${sy(front[front.length - 1].y)}"/>`;
    rows.forEach((r, i) => {
      const onFront = front.some((p) => p.r === r);
      svg += `<g class="pt" data-i="${i}"><circle cx="${sx(r[x] || 0)}" cy="${sy(r[y])}" r="7" fill="#fff" stroke="${colorFor(r)}" stroke-width="3"/>` +
        (onFront ? `<text x="${sx(r[x] || 0) + 11}" y="${sy(r[y]) - 9}" class="pt-label" fill="${colorFor(r)}">${esc(r.label)}</text>` : '') + '</g>';
    });
    el.innerHTML = svg + '</svg>';
    el.querySelectorAll('.pt').forEach((g) => {
      const r = rows[+g.dataset.i];
      bindTip(g, () => `<div class="tip-title">${tipRow(colorFor(r), `<b>${esc(r.label)}</b>`)}</div>
        <div class="tip-row">Pass rate: ${pct(r[y])}</div><div class="tip-row">${esc(xLabel)}: ${fmtX(r[x] || 0)}</div>`);
      if (onClick) g.addEventListener('click', () => onClick(r));
    });
  }

  function table(el, rows, { onClick } = {}) {
    const money = (v) => v == null ? '—' : v === 0 ? '$0' : `$${v.toFixed(v < 0.1 ? 4 : 2)}`;
    el.innerHTML = `<div class="table-wrap"><table><thead><tr>
      <th>Agent</th><th>n</th><th>Success</th><th>Reward</th><th>Process</th><th>Efficiency</th>
      <th>Budget used</th><th>Turns</th><th>Tokens</th><th>$ / run</th><th>$ / success</th></tr></thead><tbody>
      ${rows.map((r, i) => `<tr data-i="${i}"${onClick ? ' class="clickable"' : ''}>
        <td><i class="sq" style="background:${colorFor(r)}"></i><b>${esc(r.label)}</b><small>${esc(sub(r))}</small></td>
        <td>${r.n}</td><td>${pct(r.success)}</td><td>${r.reward.toFixed(3)}</td><td>${r.process.toFixed(2)}</td>
        <td>${r.efficiency.toFixed(2)}</td><td>${pct(r.budget_used)}</td><td>${r.turns.toFixed(1)}</td>
        <td>${r.tokens ? Math.round(r.tokens).toLocaleString() : '—'}</td><td>${money(r.cost_per_run)}</td><td>${money(r.cost_per_success)}</td></tr>`).join('')}
      </tbody></table></div>`;
    if (onClick) el.querySelectorAll('tr[data-i]').forEach((tr) => tr.addEventListener('click', () => onClick(rows[+tr.dataset.i])));
  }

  /* First checkpoint failure: every run follows the trunk and drops off at the first milestone it missed. */
  const CHECKPOINTS = [
    { key: 'clues',   label: 'Clues',   long: 'Read the calendar or chat before walking anywhere' },
    { key: 'floor',   label: 'Floor',   long: 'Reached the floor where the bag was' },
    { key: 'room',    label: 'Room',    long: 'Walked into the room where the bag was' },
    { key: 'search',  label: 'Search',  long: 'Searched that room' },
    { key: 'inspect', label: 'Verify',  long: "Inspected or zoomed into the owner's bag (the grader's rule)" },
  ];

  function checkpointProgress(run) {
    const tr = run.trajectory, a = run.answer, target = a.target;
    const firstMove = tr.findIndex((s) => s.tool === 'go_to');
    const read = tr.findIndex((s) => s.tool === 'read_calendar' || s.tool === 'read_chat');
    const floorOf = (n) => +String(n).slice(1, String(n).indexOf('_'));
    const nodes = tr.flatMap((s) => s.path || [s.node]);
    const met = {
      clues: read > 0 && (firstMove < 0 || read < firstMove),
      floor: nodes.some((n) => floorOf(n) === floorOf(target.node)),
      room: tr.some((s) => (s.tool === 'go_to' && !s.error && s.node === target.node) || (s.tool === 'start' && s.node === target.node)),
      search: tr.some((s) => s.tool === 'search' && !s.error && s.node === target.node),
      inspect: tr.some((s) => !s.error && ((s.tool === 'inspect' && (a.bag_objects || {})[s.args?.object] === 'target') ||
        (s.tool === 'zoom' && (target.views || [target.view]).includes(s.args?.view)))),
    };
    const skipped = CHECKPOINTS.filter((c) => !met[c.key]);
    if (run.info.success) return { met, skipped, failAt: CHECKPOINTS.length + 1 };
    const failAt = CHECKPOINTS.findIndex((c) => !met[c.key]);
    return { met, skipped, failAt: failAt >= 0 ? failAt : CHECKPOINTS.length };
  }

  function checkpoints(el, runs, { onClick, roomName = (n) => n } = {}) {
    if (!runs.length) { el.innerHTML = '<div class="empty">No runs for this task yet.</div>'; return; }
    const step = 150, W = 1190, x0 = 70, y0 = 92, rowH = 26, dropTop = 150;
    const short = (s) => { s = s.replace(/ \(.*\)$/, '').replace(' (mock)', ''); return s.length > 18 ? `${s.slice(0, 17)}…` : s; };
    const cols = CHECKPOINTS.map((_, i) => 165 + i * step);
    const xGrader = 165 + CHECKPOINTS.length * step, xPass = xGrader + 135;
    const stops = [...cols, xGrader, xPass];
    const prog = runs.map((r) => ({ r, ...checkpointProgress(r) }));
    const byStop = stops.map(() => []);
    prog.forEach((p) => byStop[p.failAt].push(p));
    const maxStack = Math.max(1, ...byStop.map((b) => b.length));
    const H = dropTop + maxStack * rowH + 30;
    const passes = byStop[stops.length - 1].length;
    let svg = `<svg viewBox="0 0 ${W} ${H}" class="cp-svg" role="img" aria-label="First checkpoint failure for ${runs.length} runs">`;
    // Grader failures zone
    if (byStop[stops.length - 2].length) svg += `<rect x="${xGrader - 34}" y="${dropTop - 22}" width="68" height="${byStop[stops.length - 2].length * rowH + 20}" rx="8" class="cp-zone"/>`;
    // Column headers and ticks
    svg += `<text x="${x0}" y="${y0 - 40}" class="cp-head" text-anchor="middle">RUNS</text>`;
    CHECKPOINTS.forEach((c, i) => {
      svg += `<g><title>${esc(c.long)}</title><text x="${cols[i]}" y="${y0 - 48}" class="cp-head" text-anchor="middle">CP${i + 1}</text>
        <text x="${cols[i]}" y="${y0 - 33}" class="cp-sub" text-anchor="middle">${c.label}</text>
        <line x1="${cols[i]}" x2="${cols[i]}" y1="${y0 - 24}" y2="${y0 + 24}" class="cp-tick"/></g>`;
    });
    svg += `<text x="${xGrader}" y="${y0 - 40}" class="cp-head" text-anchor="middle">Grader</text>
      <line x1="${xGrader}" x2="${xGrader}" y1="${y0 - 24}" y2="${y0 + 24}" class="cp-tick strong"/>
      <circle cx="${xGrader}" cy="${y0}" r="13" class="cp-ring"/>
      <text x="${xPass}" y="${y0 - 56}" class="cp-pass-count" text-anchor="middle">${passes}/${runs.length} pass</text>
      <text x="${xPass}" y="${y0 - 36}" class="cp-head" text-anchor="middle">PASS</text>
      <circle cx="${xPass}" cy="${y0}" r="24" class="cp-pass-halo"/>`;
    // Run paths
    const paths = [], dots = [];
    prog.forEach((p, i) => {
      const k = byStop[p.failAt].indexOf(p);
      const xs = stops[p.failAt], isPass = p.failAt === stops.length - 1;
      const jitter = ((i % 7) - 3) * 1.1;
      const yd = dropTop + k * rowH, xd = xs + 10 + k * 1.5;
      const bend = isPass ? xs - 6 : xs - 34;
      const c = agentColor(p.r.agent);
      paths.push(`<path d="M${x0},${y0 + jitter} H${bend} C${bend + 26},${y0 + jitter} ${xd - 8},${y0 + 30} ${xd},${yd}" class="cp-line" stroke="${c}" data-i="${i}"/>`);
      const fail = !isPass || p.skipped.length > 0;   // hollow = failed, or passed without every checkpoint
      dots.push(`<g class="cp-dot" data-i="${i}">${fail
        ? `<circle cx="${xd}" cy="${yd}" r="7" fill="#fff" stroke="${c}" stroke-width="2.2"/><circle cx="${xd}" cy="${yd}" r="2.6" fill="${c}"/>`
        : `<circle cx="${xd}" cy="${yd}" r="6.5" fill="${c}"/>`}
        <text x="${xd + 13}" y="${yd + 4}" class="cp-label" fill="${c}">${esc(short(p.r.agent.label))}</text></g>`);
    });
    svg += paths.join('');
    svg += `<circle cx="${x0}" cy="${y0}" r="14" class="cp-start-halo"/><circle cx="${x0}" cy="${y0}" r="8" class="cp-start"/>
      <text x="${x0}" y="${y0 + 36}" class="cp-sub" text-anchor="middle">${runs.length} runs</text>
      <rect x="${xPass - 11}" y="${y0 - 11}" width="22" height="22" transform="rotate(45 ${xPass} ${y0})" class="cp-diamond"/>`;
    svg += dots.join('') + '</svg>';
    el.innerHTML = svg;
    const describe = (p) => {
      const t = p.r.answer.target, room = roomName(t.node);
      const why = p.failAt < CHECKPOINTS.length ? {
        clues: 'moved before reading any clue', floor: `never reached floor ${t.node.slice(1, 2)}`,
        room: `never went to ${room}`, search: `reached ${room} but never searched it`,
        inspect: `searched ${room} but never inspected or zoomed into the right bag`,
      }[CHECKPOINTS[p.failAt].key] : p.failAt === CHECKPOINTS.length ? `reached the bag, then ${p.r.info.outcome.replace(/_/g, ' ')}`
        : p.skipped.length ? `found the right bag without: ${p.skipped.map((c) => c.label.toLowerCase()).join(', ')}` : 'found the right bag';
      return `<div class="tip-title">${tipRow(agentColor(p.r.agent), `<b>${esc(p.r.agent.label)}</b>`)}</div>
        <div class="tip-row">${p.failAt <= CHECKPOINTS.length ? `First failure: ${p.failAt < CHECKPOINTS.length ? `CP${p.failAt + 1} · ${CHECKPOINTS[p.failAt].label}` : 'Grader'}` : 'Passed'}</div>
        <div class="tip-row muted">${esc(why)}</div><div class="tip-row muted">Reward ${(+p.r.info.reward).toFixed(2)} · ${p.r.trajectory.length - 1} steps</div>`;
    };
    el.querySelectorAll('.cp-dot, .cp-line').forEach((g) => {
      const p = prog[+g.dataset.i];
      bindTip(g, () => describe(p));
      // Highlight this run's line + dot, dim the rest.
      g.addEventListener('pointerenter', () => {
        el.querySelector('svg').classList.add('focus');
        el.querySelectorAll(`[data-i="${g.dataset.i}"]`).forEach((x) => x.classList.add('on'));
      });
      g.addEventListener('pointerleave', () => {
        el.querySelector('svg').classList.remove('focus');
        el.querySelectorAll('.on').forEach((x) => x.classList.remove('on'));
      });
      if (onClick) g.addEventListener('click', () => onClick(p.r));
    });
  }

  window.Charts = { bars, stacked, pareto, table, checkpoints, checkpointProgress, CHECKPOINTS, GROUPS, colorFor, agentColor, pct };
})();
