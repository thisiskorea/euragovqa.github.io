/* ───── constants ───── */
const NATION_LABELS = ['India','EU','Japan','Taiwan','South Korea'];
const NATION_KEYS   = ['india','eu','japan','taiwan','south_korea'];
const TASK_KEYS     = [
  'biology','law','chemistry','medicine','administration','physics',
  'mathematics','computer_science','philosophy','economics','history',
  'language','geography','engineering','earth_science','psychology','politics'
];
const PALETTE = ['#1e90ff','#ff6384','#ffcd56','#4bc0c0','#9966ff',
                 '#c9cbcf','#f67019','#00a950','#c92020','#ffa500'];

let rows = [];
let dataTable = null;

/* ───── fetch & init ───── */
fetch('./data/leaderboard.json')
  .then(r=>r.json())
  .then(json=>{
    rows = json.sort((a,b)=>b.overall-a.overall);
    document.getElementById('timestamp').textContent =
      new Date().toISOString().slice(0,10);
    initTabs();
    initBar();
    initRadar();
  });

/* ───── Tabs & Table ───── */
function initTabs(){
  const tabBtns = document.querySelectorAll('.tab-btn');
  tabBtns.forEach(btn=>btn.onclick = () => {
    tabBtns.forEach(b=>b.classList.remove('active','bg-blue-600','text-white','bg-gray-200'));
    btn.classList.add('active','bg-blue-600','text-white');
    rebuildTable(btn.dataset.view);
  });
  rebuildTable('task'); // default
}

function rebuildTable(view){
  // destroy old
  if(dataTable){ dataTable.destroy(); $('#lb-table').empty(); }

  /* header */
  const thead = $('<thead><tr></tr></thead>');
  thead.find('tr').append('<th>Rank</th><th>Model</th>');
  if(view==='task'){
    TASK_KEYS.forEach(k=>thead.find('tr').append(`<th>${k}</th>`));
  }else{ // nation
    thead.find('tr').append('<th>Overall</th>');
    NATION_LABELS.forEach(n=>thead.find('tr').append(`<th>${n}</th>`));
  }
  thead.find('tr').append('<th>Paper</th>');
  $('#lb-table').append(thead);

  /* body */
  const tbody=$('<tbody></tbody>');
  rows.forEach((r,i)=>{
    const tr=$('<tr></tr>');
    tr.append(`<td>${i+1}</td><td>${r.model}</td>`);
    if(view==='task'){
      TASK_KEYS.forEach(k=>tr.append(`<td>${r.tasks[k]}</td>`));
    }else{
      tr.append(`<td>${r.overall}</td>`);
      NATION_KEYS.forEach(k=>tr.append(`<td>${r.nation[k]}</td>`));
    }
    tr.append(`<td>${r.paper?`<a href="${r.paper}" target="_blank">link</a>`:'-'}</td>`);
    tbody.append(tr);
  });
  $('#lb-table').append(tbody);

  /* DataTable */
  dataTable = $('#lb-table').DataTable({
    scrollX:true,
    order:[[ view==='task'?2:3,'desc' ]],
    pageLength:10
  });
}

/* ───── Bar Chart (metric) ───── */
function initBar(){
  const select=document.getElementById('metric-select');
  select.innerHTML =
    `<option value="overall" selected>overall</option>`+
    `<optgroup label="nation">`+
      NATION_KEYS.map((k,i)=>`<option value="${k}">${NATION_LABELS[i]}</option>`).join('')+
    `</optgroup>`+
    `<optgroup label="task">`+
      TASK_KEYS.map(t=>`<option value="${t}">${t}</option>`).join('')+
    `</optgroup>`;
  const ctx=document.getElementById('metric-chart');let chart=null;
  const draw=metric=>{
    chart&&chart.destroy();
    const labels=rows.map(r=>r.model);
    const data=metric==='overall'
      ? rows.map(r=>r.overall)
      : NATION_KEYS.includes(metric)
        ? rows.map(r=>r.nation[metric])
        : rows.map(r=>r.tasks[metric]);
    chart=new Chart(ctx,{type:'bar',
      data:{labels,datasets:[{label:metric.toUpperCase(),data}]},
      options:{responsive:true,animation:{duration:1200,easing:'easeOutQuart'},
               scales:{y:{beginAtZero:true,max:100}}}});
  };
  draw('overall');
  select.onchange=e=>draw(e.target.value);
}

/* ───── Radar (Nation) ───── */
function initRadar(){
  const modal=document.getElementById('radar-modal');
  document.getElementById('radar-btn').onclick=()=>{modal.classList.remove('hidden'); drawRadar();}
  document.getElementById('close-radar').onclick=()=>modal.classList.add('hidden');
  document.getElementById('top-n-select').onchange=drawRadar;

  let chart=null;
  function drawRadar(){
    chart&&chart.destroy();
    const topNVal=document.getElementById('top-n-select').value;
    const n = topNVal==='all'? rows.length : parseInt(topNVal,10);
    const topRows=rows.slice(0,n);
    const data={
      labels:NATION_LABELS,
      datasets: topRows.map((r,i)=>({
        label:r.model,
        data:NATION_KEYS.map(k=>r.nation[k]),
        backgroundColor:Chart.helpers.color(PALETTE[i%PALETTE.length]).alpha(0.3).rgbString(),
        borderColor:PALETTE[i%PALETTE.length], borderWidth:2, pointRadius:3, fill:true
      }))
    };
    chart=new Chart(document.getElementById('radar-canvas'),{
      type:'radar',data,
      options:{responsive:true,animation:{duration:1500,easing:'easeOutQuad'},
               scales:{r:{min:0,max:100,ticks:{stepSize:20}}}}
    });
  }
}

