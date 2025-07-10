/* ============ constants ============ */
const NATION_LABELS = ['India','EU','Japan','Taiwan','South Korea'];
const NATION_KEYS   = ['india','eu','japan','taiwan','south_korea'];
const TASK_KEYS     = [
  'biology','law','chemistry','medicine','administration','physics',
  'mathematics','computer_science','philosophy','economics','history',
  'language','geography','engineering','earth_science','psychology','politics'
];
const COLORS = ['#1e90ff','#ff6384','#ffcd56','#4bc0c0','#9966ff',
                '#c9cbcf','#f67019','#00a950','#c92020','#ffa500'];

/* ============ globals ============ */
let rows = [];
let table = null;

/* ============ helpers ============ */
const rgba = (hex,a)=>Chart.helpers.color(hex).alpha(a).rgbString();

/* ============ fetch & init ============ */
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

/* ---------- Tabs & DataTable ---------- */
function initTabs(){
  const btns=document.querySelectorAll('.tab-btn');
  btns.forEach(b=>b.onclick=()=>{
    btns.forEach(x=>x.classList.remove('active','bg-blue-600','text-white','bg-gray-200'));
    b.classList.add('active','bg-blue-600','text-white');
    buildTable(b.dataset.view);
  });
  buildTable('task');
}

function buildTable(view){
  table && table.destroy();
  $('#lb-table').empty();

  /* header */
  const headRow = [];
  headRow.push('Rank','Model');
  if(view==='task'){
    headRow.push('Overall',...TASK_KEYS);
  }else{
    headRow.push('Overall',...NATION_LABELS);
  }
  headRow.push('Paper');

  const thead=$('<thead><tr></tr></thead>');
  headRow.forEach(h=>thead.find('tr').append(`<th>${h}</th>`));
  $('#lb-table').append(thead);

  /* body */
  const tbody=$('<tbody></tbody>');
  rows.forEach((r,i)=>{
    const tr=$('<tr></tr>');
    tr.append(`<td>${i+1}</td><td>${r.model}</td>`);
    if(view==='task'){
      tr.append(`<td>${r.overall}</td>`);
      TASK_KEYS.forEach(k=>tr.append(`<td>${r.tasks[k]}</td>`));
    }else{
      tr.append(`<td>${r.overall}</td>`);
      NATION_KEYS.forEach(k=>tr.append(`<td>${r.nation[k]}</td>`));
    }
    tr.append(`<td>${r.paper?`<a href="${r.paper}" target="_blank">link</a>`:'-'}</td>`);
    tbody.append(tr);
  });
  $('#lb-table').append(tbody);

  /* DataTable init */
  table = $('#lb-table').DataTable({
    scrollX:true,
    responsive:true,
    dom:'Bfrtip',
    buttons:['colvis'],
    order:[[ view==='task'?2:3,'desc' ]],
    pageLength:10
  });
}

/* ---------- Metric Bar Chart ---------- */
function initBar(){
  const sel=document.getElementById('metric-select');
  sel.innerHTML =
    `<option value="overall">overall</option>`+
    `<optgroup label="nation">`+
      NATION_KEYS.map((k,i)=>`<option value="${k}">${NATION_LABELS[i]}</option>`).join('')+
    `</optgroup>`+
    `<optgroup label="task">`+
      TASK_KEYS.map(t=>`<option value="${t}">${t}</option>`).join('')+
    `</optgroup>`;
  const ctx=document.getElementById('metric-chart');
  let chart=null;
  const draw=metric=>{
    chart&&chart.destroy();
    const labels=rows.map(r=>r.model);
    const data =
      metric==='overall' ? rows.map(r=>r.overall) :
      NATION_KEYS.includes(metric) ? rows.map(r=>r.nation[metric]) :
      rows.map(r=>r.tasks[metric]);
    chart=new Chart(ctx,{type:'bar',
      data:{labels,datasets:[{label:metric.toUpperCase(),data}]},
      options:{responsive:true,animation:{duration:1200,easing:'easeOutQuart'},
               scales:{y:{min:0,max:100}}}});
  };
  draw('overall');
  sel.onchange=e=>draw(e.target.value);
}

/* ---------- Nation Radar Chart ---------- */
function initRadar(){
  const modal = document.getElementById('radar-modal');
  const radarBtn=document.getElementById('radar-btn');
  const closeBtn=document.getElementById('close-radar');
  const topSel=document.getElementById('top-n-select');
  const ctx=document.getElementById('radar-canvas');
  let chart=null;

  const draw = () => {
    chart&&chart.destroy();
    const val=topSel.value;
    const n= val==='all'? rows.length : parseInt(val,10);
    const tops=rows.slice(0,n);
    const data={
      labels:NATION_LABELS,
      datasets: tops.map((r,i)=>({
        label:r.model,
        data:NATION_KEYS.map(k=>r.nation[k]),
        backgroundColor:rgba(COLORS[i%COLORS.length],0.3),
        borderColor:COLORS[i%COLORS.length],
        fill:true,borderWidth:2,pointRadius:3
      }))
    };
    chart=new Chart(ctx,{type:'radar',data,
      options:{responsive:true,animation:{duration:1500,easing:'easeOutQuad'},
               scales:{r:{min:0,max:100,ticks:{stepSize:20}}}}});
  };

  radarBtn.onclick = ()=>{ modal.classList.remove('hidden'); draw(); };
  closeBtn.onclick = ()=> modal.classList.add('hidden');
  topSel.onchange = draw;
}
