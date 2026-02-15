/**
 * EuraGovQA Leaderboard & Charts
 * Clean, minimal academic style
 */

const NATIONS = ['india', 'eu', 'japan', 'taiwan', 'south_korea'];
const NATION_LABELS = ['India', 'EU', 'Japan', 'Taiwan', 'S.Korea'];
const SUBJECTS = [
  'biology', 'law', 'chemistry', 'medicine', 'administration', 'physics',
  'mathematics', 'computer_science', 'philosophy', 'economics', 'history',
  'language', 'geography', 'engineering', 'earth_science', 'psychology', 'politics'
];

const COLORS = {
  amber: '#f59e0b',
  slate: '#64748b',
  emerald: '#10b981',
  blue: '#3b82f6',
  red: '#ef4444',
  purple: '#8b5cf6',
};

let data = [];

// Load data and initialize
fetch('./data/leaderboard.json')
  .then(r => r.json())
  .then(json => {
    data = json.sort((a, b) => b.overall - a.overall);
    initLeaderboard();
    initCharts();
  })
  .catch(err => console.error('Failed to load leaderboard data:', err));

/**
 * Leaderboard Table
 */
function initLeaderboard() {
  renderTable('overall');

  // Tab switching
  document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      renderTable(btn.dataset.view);
    });
  });
}

function renderTable(view) {
  const tbody = document.getElementById('lb-body');
  const thead = document.querySelector('#lb-table thead tr');

  // Update headers based on view
  let headers = ['Rank', 'Model', 'Overall'];
  if (view === 'overall' || view === 'nation') {
    headers = headers.concat(NATION_LABELS);
  } else if (view === 'task') {
    headers = headers.concat(SUBJECTS.slice(0, 8).map(s => s.charAt(0).toUpperCase() + s.slice(1)));
  }
  headers.push('Paper');

  thead.innerHTML = headers.map((h, i) =>
    `<th class="${i === 1 ? '' : 'center'}">${h}</th>`
  ).join('');

  // Render rows
  tbody.innerHTML = data.map((row, i) => {
    const rank = i + 1;
    const rankClass = rank <= 3 ? `rank-${rank}` : '';
    const medal = rank === 1 ? '🥇' : rank === 2 ? '🥈' : rank === 3 ? '🥉' : rank;

    let cells = `
      <td class="center"><span class="medal">${medal}</span></td>
      <td><span class="model-name">${row.model}</span></td>
      <td class="center"><span class="score ${getScoreClass(row.overall)}">${row.overall.toFixed(1)}</span></td>
    `;

    if (view === 'overall' || view === 'nation') {
      NATIONS.forEach(n => {
        const score = row.nation[n];
        cells += `<td class="center"><span class="score ${getScoreClass(score)}">${score.toFixed(1)}</span></td>`;
      });
    } else if (view === 'task') {
      SUBJECTS.slice(0, 8).forEach(s => {
        const score = row.tasks[s];
        cells += `<td class="center"><span class="score ${getScoreClass(score)}">${score.toFixed(1)}</span></td>`;
      });
    }

    cells += `<td class="center">${row.paper ? `<a href="${row.paper}" target="_blank" class="paper-link">Link</a>` : '—'}</td>`;

    return `<tr class="${rankClass}">${cells}</tr>`;
  }).join('');
}

function getScoreClass(score) {
  if (score >= 80) return 'high';
  if (score >= 60) return 'mid';
  return 'low';
}

/**
 * Charts
 */
function initCharts() {
  // Overview radar chart
  const radarCtx = document.getElementById('overview-radar');
  if (radarCtx) {
    new Chart(radarCtx, {
      type: 'radar',
      data: {
        labels: ['India', 'EU', 'Japan', 'Taiwan', 'S.Korea'],
        datasets: data.slice(0, 3).map((row, i) => ({
          label: row.model,
          data: NATIONS.map(n => row.nation[n]),
          borderColor: [COLORS.amber, COLORS.blue, COLORS.emerald][i],
          backgroundColor: [
            'rgba(245, 158, 11, 0.1)',
            'rgba(59, 130, 246, 0.1)',
            'rgba(16, 185, 129, 0.1)'
          ][i],
          borderWidth: 2,
          pointRadius: 3,
          pointBackgroundColor: [COLORS.amber, COLORS.blue, COLORS.emerald][i],
        }))
      },
      options: {
        responsive: true,
        plugins: {
          legend: {
            position: 'bottom',
            labels: {
              usePointStyle: true,
              padding: 16,
              font: { size: 11 }
            }
          }
        },
        scales: {
          r: {
            min: 50,
            max: 100,
            ticks: {
              stepSize: 10,
              font: { size: 10 },
              backdropColor: 'transparent'
            },
            pointLabels: {
              font: { size: 11, weight: '500' }
            },
            grid: { color: '#e2e8f0' },
            angleLines: { color: '#e2e8f0' }
          }
        }
      }
    });
  }

  // Country distribution (doughnut)
  const countryCtx = document.getElementById('country-chart');
  if (countryCtx) {
    new Chart(countryCtx, {
      type: 'doughnut',
      data: {
        labels: ['India', 'EU', 'Japan', 'Taiwan', 'South Korea'],
        datasets: [{
          data: [4521, 2134, 2876, 2543, 3160],
          backgroundColor: [
            '#f97316', '#3b82f6', '#ef4444', '#10b981', '#8b5cf6'
          ],
          borderWidth: 0,
          hoverOffset: 8
        }]
      },
      options: {
        responsive: true,
        cutout: '60%',
        plugins: {
          legend: {
            position: 'bottom',
            labels: {
              usePointStyle: true,
              padding: 16,
              font: { size: 11 }
            }
          }
        }
      }
    });
  }

  // Subject distribution (bar)
  const subjectCtx = document.getElementById('subject-chart');
  if (subjectCtx) {
    const subjectData = [
      { name: 'Law', count: 1423 },
      { name: 'Admin', count: 1287 },
      { name: 'Econ', count: 1156 },
      { name: 'History', count: 1089 },
      { name: 'Geo', count: 1034 },
      { name: 'Math', count: 987 },
      { name: 'Physics', count: 923 },
      { name: 'Chem', count: 876 },
      { name: 'Bio', count: 845 },
      { name: 'CS', count: 812 },
    ];

    new Chart(subjectCtx, {
      type: 'bar',
      data: {
        labels: subjectData.map(d => d.name),
        datasets: [{
          data: subjectData.map(d => d.count),
          backgroundColor: '#0f172a',
          borderRadius: 4,
          barThickness: 20
        }]
      },
      options: {
        responsive: true,
        indexAxis: 'y',
        plugins: {
          legend: { display: false }
        },
        scales: {
          x: {
            grid: { color: '#e2e8f0' },
            ticks: { font: { size: 10 } }
          },
          y: {
            grid: { display: false },
            ticks: { font: { size: 11 } }
          }
        }
      }
    });
  }
}
