/* === constants === */
const NATION_LABELS=['India','EU','Japan','Taiwan','South Korea'];
const NATION_KEYS  =['india','eu','japan','taiwan','south_korea'];
const TASK_KEYS=[
  'biology','law','chemistry','medicine','administration','physics',
  'mathematics','computer_science','philosophy','economics','history',
  'language','geography','engineering','earth_science','psychology','politics'];
const COLORS=['#1e90ff','#ff6384','#ffcd56','#4bc0c0','#9966ff',
              '#c9cbcf','#f67019','#00a950','#c92020','#ffa500'];

let rows=[],table=null;

/* === fetch === */
fetch('./data/leaderboard.json').then(r=>r.json()).then(json=>{
  rows=json.sort((a,b)=>b.overall-a.overall);
  document.getElementById('timestamp').textContent=new Date().toISOString().slice(0,10);
  initTabs();initBar();initRadar();
});

/* === tabs & table === */
function initTabs(){
  document.querySelectorAll('.tab-btn').forEach(btn=>{
    btn.onclick=()=>{document.querySelectorAll('.tab-btn').forEach(b=>b.classList.remove('bg-blue-600','text-white','active'));
      btn.classList.add('bg-blue-600','text-white','active');
      buildTable(btn.dataset.view);};});
  buildTable('task');
}

function buildTable(view){
  table&&table.destroy();$('#lb-table').empty();
  const headers=['Rank','Model',...(view==='task'?['Overall',...TASK_KEYS]:['Overall',...NATION_LABELS]),'Paper'];
  $('#lb-table').append('<thead><tr>'+headers.map(h=>`<th>${h}</th>`).join('')+'</tr></thead>');
  const tbody=$('<tbody></tbody>');
  rows.forEach((r,i)=>{
    const cells=[i+1,r.model];
    if(view==='task'){cells.push(r.overall);TASK_KEYS.forEach(k=>cells.push(r.tasks[k]));}
    else{cells.push(r.overall);NATION_KEYS.forEach(k=>cells.push(r.nation[k]));}
    cells.push(r.paper?`<a href="${r.paper}" target="_blank">link</a>`:'-');
    tbody.append('<tr>'+cells.map(c=>`<td>${c}</td>`).join('')+'</tr>');
  });
  $('#lb-table').append(tbody);
  table=$('#lb-table').DataTable({
    scrollX:true,responsive:true,dom:'Bfrtip',buttons:['colvis'],
    columnDefs:[{targets:'_all',className:'dt-center'}],
    order:[[view==='task'?2:3,'desc']],pageLength:10});
}

/* === bar chart === */
function initBar(){
  const sel=document.getElementById('metric-select');
  sel.innerHTML='<option value="overall">overall</option>'+
    '<optgroup label="nation">'+NATION_KEYS.map((k,i)=>`<option value="${k}">${NATION_LABELS[i]}</option>`).join('')+'</optgroup>'+
    '<optgroup label="task">'+TASK_KEYS.map(t=>`<option value="${t}">${t}</option>`).join('')+'</optgroup>';
  const ctx=document.getElementById('metric-chart');let chart=null;
  const draw=m=>{
    chart&&chart.destroy();
    const data=m==='overall'?rows.map(r=>r.overall):
      NATION_KEYS.includes(m)?rows.map(r=>r.nation[m]):rows.map(r=>r.tasks[m]);
    chart=new Chart(ctx,{type:'bar',data:{labels:rows.map(r=>r.model),datasets:[{label:m.toUpperCase(),data}]},
      options:{animation:{duration:1200,easing:'easeOutQuart'},scales:{y:{min:0,max:100}}}});
  };
  draw('overall');sel.onchange=e=>draw(e.target.value);
}

/* === radar === */
function initRadar(){
  const modal=document.getElementById('radar-modal'),ctx=document.getElementById('radar-canvas');
  document.getElementById('radar-btn').onclick=()=>{modal.classList.remove('hidden');draw();}
  document.getElementById('close-radar').onclick=()=>modal.classList.add('hidden');
  document.getElementById('top-n-select').onchange=draw;
  let chart=null;
  function draw(){
    chart&&chart.destroy();
    const nSel=document.getElementById('top-n-select').value;
    const tops=nSel==='all'?rows:rows.slice(0,parseInt(nSel));
    chart=new Chart(ctx,{type:'radar',
      data:{labels:NATION_LABELS,
            datasets:tops.map((r,i)=>({label:r.model,data:NATION_KEYS.map(k=>r.nation[k]),
              backgroundColor:Chart.helpers.color(COLORS[i%COLORS.length]).alpha(0.3).rgbString(),
              borderColor:COLORS[i%COLORS.length],fill:true,borderWidth:2,pointRadius:3}))},
      options:{responsive:true,animation:{duration:1500,easing:'easeOutQuad'},
               scales:{r:{min:0,max:100,ticks:{stepSize:20}}}}});
  }
}
