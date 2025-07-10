/* Leaderboard + Metric 그래프 */
const nations = ['india','eu','japan','taiwan','south_korea'];
const tasks   = [
  'biology','law','chemistry','medicine','administration','physics',
  'mathematics','computer_science','philosophy','economics','history',
  'language','geography','engineering','earth_science','psychology','politics'
];

fetch('./data/leaderboard.json')
  .then(r => r.json())
  .then(rows => {
    /* ---------- DataTable ---------- */
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
    $('#lb-table').DataTable({ order: [[2,'desc']], pageLength: 10 });
    document.getElementById('timestamp').textContent =
      new Date().toISOString().slice(0,10);

    /* ---------- Metric dropdown + Chart ---------- */
    const select = document.getElementById('metric-select');
    select.innerHTML =
      `<option value="overall">overall</option>` +
      `<optgroup label="nation">` +
        nations.map(n => `<option value="${n}">${n}</option>`).join('') +
      `</optgroup>` +
      `<optgroup label="task">` +
        tasks.map(t => `<option value="${t}">${t}</option>`).join('') +
      `</optgroup>`;

    const ctx = document.getElementById('metric-chart');
    let chart = null;
    const draw = metric => {
      if (chart) chart.destroy();
      const labels = rows.map(r => r.model);
      let data;
      if (metric === 'overall') {
        data = rows.map(r => r.overall);
      } else if (nations.includes(metric)) {
        data = rows.map(r => r.nation[metric] ?? 0);
      } else { // task
        data = rows.map(r => r.tasks[metric] ?? 0);
      }
      chart = new Chart(ctx, {
        type: 'bar',
        data: { labels,
                datasets: [{ label: metric.toUpperCase(), data }] },
        options: { responsive:true,
                   scales:{ y:{ beginAtZero:true, max:100 } } }
      });
    };

    draw('overall');
    select.addEventListener('change', e => draw(e.target.value));
  })
  .catch(err => console.error(err));
