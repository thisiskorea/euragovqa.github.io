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
      tr.app

}

