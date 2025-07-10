/* ------------------------------ constants ------------------------------ */
const NATION_LABELS = ['India', 'EU', 'Japan', 'Taiwan', 'South Korea'];
const NATION_KEYS   = ['india','eu','japan','taiwan','south_korea'];

const TASK_KEYS = [
  'biology','law','chemistry','medicine','administration','physics',
  'mathematics','computer_science','philosophy','economics','history',
  'language','geography','engineering','earth_science','psychology','politics'
];

const BAR_ANIM_MS = 1200; // 바차트 애니메이션(ms)

/* ------------------------------ helpers ----------------------------- */
const makeColor = (hex, alpha) =>
  Chart.helpers.color(hex).alpha(alpha).rgbString();

/* ------------------------------ main ----------------------------- */
fetch('./data/leaderboard.json')
  .then(r => r.json())
  .then(rows => {
    /* ---------- TABLE ---------- */
    rows.sort((a,b)=>b.overall-a.overall);
    const tbody = $('#lb-table tbody');
    rows.forEach((r,i)=>{
      tbody.append(`
        <tr>
          <td>${i+1}</td>
          <td>${r.model}</td>
          <td>${r.overall}</td>
          <td>${r.tasks.law}</td>
          <td>${r.tasks.biology}</td>
          <td>${r.paper ? `<a href="${r.paper}" target="_blank" class="text-blue-600 underline">paper</a>` : '-'}</td>
        </tr>
      `);
    });
    $('#lb-table').DataTable({order:[[2,'desc']],pageLength:10});
    document.getElementById('timestamp').textContent =
      new Date().toISOString().slice(0,10);

    /* ---------- BAR ---------- */
    const select = document.getElementById('metric-select');
    select.innerHTML =
      `<option value="overall" selected>overall</option>` +
      `<optgroup label="nation">` +
        NATION_KEYS.map((k,i)=>`<option value="${k}">${NATION_LABELS[i]}</option>`).join('') +
      `</optgroup>` +
      `<optgroup label="task">` +
        TASK_KEYS.map(t=>`<option value="${t}">${t}</option>`).join('') +
      `</optgroup>`;

    const barCtx=document.getElementById('metric-chart');let barChart=null;
    const drawBar=metric=>{
      barChart&&barChart.destroy();
      const labels=rows.map(r=>r.model);
      const data=metric==='overall'
        ? rows.map(r=>r.overall)
        : NATION_KEYS.includes(metric)
          ? rows.map(r=>r.nation?.[metric]||0)
          : rows.map(r=>r.tasks?.[metric]||0);
      barChart=new Chart(barCtx,{
        type:'bar',
        data:{labels,datasets:[{label:metric.toUpperCase(),data}]},
        options:{responsive:true,
                 animation:{duration:BAR_ANIM_MS,easing:'easeOutQuart'},
                 scales:{y:{beginAtZero:true,max:100}}}
      });
    };
    drawBar('overall');
    select.addEventListener('change',e=>drawBar(e.target.value));

    /* ---------- NATION RADAR ---------- */
    const modal   = document.getElementById('radar-modal');
    const radarBtn= document.getElementById('radar-btn');
    const closeBtn= document.getElementById('close-radar');
    const topSel  = document.getElementById('top-n-select');
    const radarCtx= document.getElementById('radar-canvas');
    const palette = ['#1e90ff','#ff6384','#ffcd56','#4bc0c0','#9966ff',
                     '#c9cbcf','#f67019','#00a950','#c92020','#ffa500'];

    let radarChart=null;
    const drawRadar = topN => {
      radarChart&&radarChart.destroy();
      const n = topN==='all'? rows.length : parseInt(topN,10);
      const topRows = rows.slice(0,n);
      const data = {
        labels:NATION_LABELS,
        datasets: topRows.map((r,i)=>({
          label:r.model,
          data:NATION_KEYS.map(k=>r.nation?.[k]||0),
          backgroundColor:makeColor(palette[i%palette.length],0.4),
          borderColor:palette[i%palette.length],
          borderWidth:2,
          pointRadius:3,
          fill:true
        }))
      };
      radarChart=new Chart(radarCtx,{
        type:'radar',
        data,
        options:{
          responsive:true,
          animation:{duration:1500,easing:'easeOutQuad'},
          scales:{r:{min:0,max:100,ticks:{stepSize:20}}},
          plugins:{legend:{position:'top'}}
        }
      });
    };

    radarBtn.onclick   = ()=>{ modal.classList.remove('hidden'); drawRadar(topSel.value); };
    closeBtn.onclick   = ()=> modal.classList.add('hidden');
    topSel.onchange    = e  => drawRadar(e.target.value);
  })
  .catch(console.error);
