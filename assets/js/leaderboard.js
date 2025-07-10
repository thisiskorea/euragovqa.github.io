// assets/js/leaderboard.js
let rowsGlobal = [];
const nations  = ['India','EU','Japan','Taiwan','South Korea'];
const tasks    = ['biology','law','chemistry','medicine','administration','physics',
                  'mathematics','computer_science','philosophy','economics','history',
                  'language','geography','engineering','earth_science','psychology','politics'];

fetch('./data/leaderboard.json')
  .then(r => r.json())
  .then(rows => {
    // ---- 1. DataTable -----------------------------------------------------------------
    rowsGlobal = rows;
    rows.sort((a, b) => b.overall - a.overall);
    const tbody = $('#lb-table tbody');
    rows.forEach((r, idx) => {
      tbody.append(`
        <tr>
          <td>${idx + 1}</td>
          <td>${r.model}</td>
          <td>${r.overall}</td>
          <td>${r.law}</td>
          <td>${r.biology}</td>
          <td>${r.paper ? `<a href="${r.paper}" target="_blank" class="text-blue-600 underline">paper</a>` : '-'}</td>
        </tr>
      `);
    });
    $('#lb-table').DataTable({ order: [[2,'desc']], pageLength: 10 });

    // timestamp
    document.getElementById('timestamp').textContent =
      new Date().toISOString().slice(0,10);

    // ---- 2. Metric dropdown  -----------------------------------------------------------
    const select = document.getElementById('metric-select');
    select.innerHTML = `
      <option value="overall" selected>Overall</option>
      <optgroup label="Nation">
        ${nations.map(n => `<option value="${n.toLowerCase().replace(/\\s/g,'_')}">${n}</option>`).join('')}
      </optgroup>
      <optgroup label="Task">
        ${tasks.map(t => `<option value="${t}">${t}</option>`).join('')}
      </optgroup>
    `;

    // ---- 3. Chart ----------------------------------------------------------------------
    const ctx   = document.getElementById('metric-chart');
    let chart   = null;
    const makeChart = metric => {
      const labels = rowsGlobal.map(r => r.model);
      const data   = rowsGlobal.map(r => r[metric] ?? (r.nation?.[metric] || r.tasks?.[metric] || 0));
      if (chart) chart.destroy();
      chart = new Chart(ctx, {
        type: 'bar',
        data: { labels,
                datasets: [{ label: metric.toUpperCase(), data }] },
        options: { responsive: true,
                   scales:{ y:{ beginAtZero:true, max:100 } } }
      });
    };
    makeChart('overall');
    select.addEventListener('change', e => makeChart(e.target.value));
  })
  .catch(err => console.error(err));
