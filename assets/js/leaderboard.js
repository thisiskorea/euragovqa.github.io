// assets/js/leaderboard.js
fetch('data/leaderboard.json')
  .then(r => r.json())
  .then(rows => {
    // sort by overall desc
    rows.sort((a, b) => b.overall - a.overall);

    // build table rows
    const tbody = $('#lb-table tbody');
    rows.forEach((r, idx) => {
      tbody.append(`
        <tr>
          <td>${idx + 1}</td>
          <td>${r.model}</td>
          <td>${r.overall}</td>
          <td>${r.law}</td>
          <td>${r.biology}</td>
          <td>
            ${r.paper ? `<a href="${r.paper}" target="_blank" class="text-blue-600 underline">paper</a>` : '-'}
          </td>
        </tr>
      `);
    });

    // init DataTables
    $('#lb-table').DataTable({
      order: [[2, 'desc']],
      pageLength: 10
    });

    // timestamp
    document.getElementById('timestamp').textContent =
      new Date().toISOString().slice(0, 10);
  });
