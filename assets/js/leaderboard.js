/* ------------------------------ constants ------------------------------ */
const nations = ['india','eu','japan','taiwan','south_korea'];
const tasks   = [
  'biology','law','chemistry','medicine','administration','physics',
  'mathematics','computer_science','philosophy','economics','history',
  'language','geography','engineering','earth_science','psychology','politics'
];

/* ------------------------------ main logic ----------------------------- */
fetch('./data/leaderboard.json')
  .then(r => r.json())
  .then(rows => {
    /* ---------- build DataTable ---------- */
    rows.sort((a, b) => b.overall - a.overall);
    const tbody = $('#lb-table tbody');
    rows.forEach((r, idx) => {
      tbody.append(`
        <tr>
          <td>${idx + 1}</td>
          <td>${r.model}</td>
          <td>${r.overall}</td>
          <td>${r.tasks.law}</td>
          <td>${r.tasks.biology}</td>
          <td>${r.paper ? `<a href="${r.paper}" target="_blank" class="text-blue-600 underline">paper</a>` : '-'}</td>
        </tr>
      `);
    });
    $('#lb-table').DataTable({ order: [[2, 'desc']], pageLength: 10 });
    document.getElementById('timestamp').textContent =
      new Date().toISOString().slice(0, 10);

    /* ---------- dropdown bar chart ---------- */
    const select = document.getElementById('metric-select');
    select.innerHTML =
      `<option value="overall">overall</option>` +
      `<optgroup label="nation">` +
        nations.map(n => `<option value="${n}">${n}</option>`).join('') +
      `</optgroup>` +
      `<optgroup label="task">` +
        tasks.map(t => `<option value="${t}">${t}</option>`).join('') +
      `</optgroup>`;

    const barCtx = document.getElementById('metric-chart');
    let barChart = null;
    const drawBar = metric => {
      if (barChart) barChart.destroy();
      const labels = rows.map(r => r.model);
      const data = metric === 'overall'
        ? rows.map(r => r.overall)
        : nations.includes(metric)
          ? rows.map(r => r.nation?.[metric] ?? 0)
          : rows.map(r => r.tasks?.[metric] ?? 0);

      barChart = new Chart(barCtx, {
        type: 'bar',
        data: { labels,
                datasets: [{ label: metric.toUpperCase(), data }] },
        options: { responsive:true, scales:{ y:{ beginAtZero:true, max:100 } } }
      });
    };
    drawBar('overall');
    select.addEventListener('change', e => drawBar(e.target.value));

    /* -------------------- Radar Chart (Top-3 models) -------------------- */
    const buckets = {
      administration: ['administration','politics','law'],
      humanities:     ['history','language','philosophy'],
      social:         ['economics','geography','psychology'],
      stem:           ['mathematics','physics','chemistry','engineering',
                       'computer_science','earth_science','biology','medicine']
    };

    // show & hide modal
    const modal   = document.getElementById('radar-modal');
    document.getElementById('radar-btn').onclick  = () => modal.classList.remove('hidden');
    document.getElementById('close-radar').onclick = () => modal.classList.add('hidden');

    const radarCtx = document.getElementById('radar-canvas');
    const colors = [
      'rgba(30,144,255,0.5)',
      'rgba(255,99,132,0.5)',
      'rgba(255,205,86,0.5)'
    ];
    const top3 = rows.slice(0, 3);
    const radarData = {
      labels: Object.keys(buckets),
      datasets: top3.map((r, i) => ({
        label: r.model,
        data: Object.values(buckets).map(arr =>
          arr.reduce((sum, k) => sum + (r.tasks[k] || 0), 0) / arr.length),
        backgroundColor: colors[i],
        borderColor: colors[i].replace('0.5', '1'),
        borderWidth: 1,
        fill: true
      }))
    };
    new Chart(radarCtx, {
      type: 'radar',
      data: radarData,
      options: {
        responsive: true,
        scales: { r: { beginAtZero: true, max: 100 } },
        plugins: { legend: { position: 'top' } }
      }
    });
  })
  .catch(err => console.error(err));

