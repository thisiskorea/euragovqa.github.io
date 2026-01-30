/* === constants === */
const NATION_LABELS = ['India', 'EU', 'Japan', 'Taiwan', 'South Korea'];
const NATION_KEYS = ['india', 'eu', 'japan', 'taiwan', 'south_korea'];
const TASK_KEYS = [
  'biology', 'law', 'chemistry', 'medicine', 'administration', 'physics',
  'mathematics', 'computer_science', 'philosophy', 'economics', 'history',
  'language', 'geography', 'engineering', 'earth_science', 'psychology', 'politics'
];
const TASK_LABELS = [
  'Biology', 'Law', 'Chemistry', 'Medicine', 'Administration', 'Physics',
  'Mathematics', 'Computer Science', 'Philosophy', 'Economics', 'History',
  'Language', 'Geography', 'Engineering', 'Earth Science', 'Psychology', 'Politics'
];
const COLORS = [
  '#3b82f6', '#ef4444', '#f59e0b', '#10b981', '#8b5cf6',
  '#6366f1', '#ec4899', '#14b8a6', '#f97316', '#06b6d4',
  '#84cc16', '#a855f7', '#22c55e'
];

let rows = [], table = null;

/* === fetch === */
fetch('./data/leaderboard.json')
  .then(r => r.json())
  .then(json => {
    rows = json.sort((a, b) => b.overall - a.overall);
    document.getElementById('timestamp').textContent = new Date().toISOString().slice(0, 10);
    initTabs();
    initBar();
    initRadar();
  });

/* === tabs & table === */
function initTabs() {
  document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.onclick = () => {
      document.querySelectorAll('.tab-btn').forEach(b => {
        b.classList.remove('bg-white', 'text-blue-600', 'shadow', 'active');
        b.classList.add('text-gray-600');
      });
      btn.classList.remove('text-gray-600');
      btn.classList.add('bg-white', 'text-blue-600', 'shadow', 'active');
      buildTable(btn.dataset.view);
    };
  });
  buildTable('task');
}

function buildTable(view) {
  if (table) {
    table.destroy();
  }
  $('#lb-table').empty();

  const isTask = view === 'task';
  const headers = [
    'Rank',
    'Model',
    'Overall',
    ...(isTask ? TASK_LABELS : NATION_LABELS),
    'Paper'
  ];

  $('#lb-table').append('<thead><tr>' + headers.map(h => `<th>${h}</th>`).join('') + '</tr></thead>');

  const tbody = $('<tbody></tbody>');
  rows.forEach((r, i) => {
    const cells = [
      getMedalRank(i + 1),
      `<span class="font-medium">${r.model}</span>`
    ];

    // Overall score with color coding
    cells.push(getScoreCell(r.overall));

    if (isTask) {
      TASK_KEYS.forEach(k => cells.push(getScoreCell(r.tasks[k])));
    } else {
      NATION_KEYS.forEach(k => cells.push(getScoreCell(r.nation[k])));
    }

    cells.push(r.paper
      ? `<a href="${r.paper}" target="_blank" class="text-blue-600 hover:text-blue-800 hover:underline">Link</a>`
      : '-');

    tbody.append('<tr>' + cells.map(c => `<td>${c}</td>`).join('') + '</tr>');
  });

  $('#lb-table').append(tbody);

  table = $('#lb-table').DataTable({
    scrollX: true,
    responsive: true,
    dom: 'Bfrtip',
    buttons: ['colvis'],
    columnDefs: [{ targets: '_all', className: 'dt-center' }],
    order: [[2, 'desc']],
    pageLength: 15,
    language: {
      search: "Search models:",
      lengthMenu: "Show _MENU_ models",
      info: "Showing _START_ to _END_ of _TOTAL_ models"
    }
  });
}

function getMedalRank(rank) {
  if (rank === 1) return '<span class="text-xl">🥇</span>';
  if (rank === 2) return '<span class="text-xl">🥈</span>';
  if (rank === 3) return '<span class="text-xl">🥉</span>';
  return `<span class="text-gray-600">${rank}</span>`;
}

function getScoreCell(score) {
  let colorClass = 'text-gray-700';
  if (score >= 80) colorClass = 'text-emerald-600 font-semibold';
  else if (score >= 70) colorClass = 'text-blue-600';
  else if (score >= 60) colorClass = 'text-amber-600';
  else if (score < 50) colorClass = 'text-red-500';
  return `<span class="${colorClass}">${score.toFixed(1)}</span>`;
}

/* === bar chart === */
function initBar() {
  const sel = document.getElementById('metric-select');
  sel.innerHTML = '<option value="overall">Overall</option>' +
    '<optgroup label="By Nation">' +
    NATION_KEYS.map((k, i) => `<option value="${k}">${NATION_LABELS[i]}</option>`).join('') +
    '</optgroup>' +
    '<optgroup label="By Task">' +
    TASK_KEYS.map((t, i) => `<option value="${t}">${TASK_LABELS[i]}</option>`).join('') +
    '</optgroup>';

  const ctx = document.getElementById('metric-chart');
  let chart = null;

  const draw = m => {
    if (chart) chart.destroy();

    const data = m === 'overall' ? rows.map(r => r.overall) :
      NATION_KEYS.includes(m) ? rows.map(r => r.nation[m]) : rows.map(r => r.tasks[m]);

    const backgroundColors = rows.map((_, i) => COLORS[i % COLORS.length]);

    chart = new Chart(ctx, {
      type: 'bar',
      data: {
        labels: rows.map(r => r.model),
        datasets: [{
          label: m.charAt(0).toUpperCase() + m.slice(1).replace('_', ' '),
          data,
          backgroundColor: backgroundColors,
          borderColor: backgroundColors.map(c => c),
          borderWidth: 1,
          borderRadius: 4
        }]
      },
      options: {
        animation: { duration: 800, easing: 'easeOutQuart' },
        responsive: true,
        maintainAspectRatio: true,
        plugins: {
          legend: { display: false },
          tooltip: {
            backgroundColor: '#1e293b',
            titleColor: '#f8fafc',
            bodyColor: '#e2e8f0',
            padding: 12,
            cornerRadius: 8,
            callbacks: {
              label: ctx => `Score: ${ctx.raw.toFixed(1)}%`
            }
          }
        },
        scales: {
          y: {
            min: 0,
            max: 100,
            grid: { color: '#e2e8f0' },
            ticks: { callback: v => v + '%' }
          },
          x: {
            grid: { display: false },
            ticks: {
              maxRotation: 45,
              minRotation: 45,
              font: { size: 11 }
            }
          }
        }
      }
    });
  };

  draw('overall');
  sel.onchange = e => draw(e.target.value);
}

/* === radar === */
function initRadar() {
  const modal = document.getElementById('radar-modal');
  const ctx = document.getElementById('radar-canvas');

  document.getElementById('radar-btn').onclick = () => {
    modal.classList.remove('hidden');
    modal.classList.add('flex');
    draw();
  };

  document.getElementById('close-radar').onclick = () => {
    modal.classList.add('hidden');
    modal.classList.remove('flex');
  };

  // Close on backdrop click
  modal.onclick = e => {
    if (e.target === modal) {
      modal.classList.add('hidden');
      modal.classList.remove('flex');
    }
  };

  document.getElementById('top-n-select').onchange = draw;

  let chart = null;

  function draw() {
    if (chart) chart.destroy();

    const nSel = document.getElementById('top-n-select').value;
    const tops = nSel === 'all' ? rows : rows.slice(0, parseInt(nSel));

    chart = new Chart(ctx, {
      type: 'radar',
      data: {
        labels: NATION_LABELS,
        datasets: tops.map((r, i) => ({
          label: r.model,
          data: NATION_KEYS.map(k => r.nation[k]),
          backgroundColor: hexToRgba(COLORS[i % COLORS.length], 0.15),
          borderColor: COLORS[i % COLORS.length],
          fill: true,
          borderWidth: 2,
          pointRadius: 4,
          pointBackgroundColor: COLORS[i % COLORS.length],
          pointBorderColor: '#fff',
          pointBorderWidth: 1
        }))
      },
      options: {
        responsive: true,
        animation: { duration: 1000, easing: 'easeOutQuad' },
        plugins: {
          legend: {
            position: 'bottom',
            labels: {
              padding: 20,
              usePointStyle: true,
              pointStyle: 'circle'
            }
          },
          tooltip: {
            backgroundColor: '#1e293b',
            titleColor: '#f8fafc',
            bodyColor: '#e2e8f0',
            padding: 12,
            cornerRadius: 8
          }
        },
        scales: {
          r: {
            min: 0,
            max: 100,
            ticks: {
              stepSize: 20,
              backdropColor: 'transparent',
              font: { size: 10 }
            },
            grid: { color: '#e2e8f0' },
            angleLines: { color: '#e2e8f0' },
            pointLabels: {
              font: { size: 12, weight: '500' },
              color: '#374151'
            }
          }
        }
      }
    });
  }
}

function hexToRgba(hex, alpha) {
  const r = parseInt(hex.slice(1, 3), 16);
  const g = parseInt(hex.slice(3, 5), 16);
  const b = parseInt(hex.slice(5, 7), 16);
  return `rgba(${r}, ${g}, ${b}, ${alpha})`;
}
