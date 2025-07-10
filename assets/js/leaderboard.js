fetch('data/leaderboard.json')
  .then(r => r.json())
  .then(rows => {
    const tbody = document.getElementById('lb-body');
    rows.forEach(r => {
      const tr = document.createElement('tr');
      tr.innerHTML = `
        <td class="border px-4 py-2 font-medium">${r.model}</td>
        <td class="border px-4 py-2 text-center">${r.overall}</td>
        <td class="border px-4 py-2 text-center">${r.law}</td>
        <td class="border px-4 py-2 text-center">${r.biology}</td>
      `;
      tbody.appendChild(tr);
    });
  });
