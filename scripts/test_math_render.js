// 渲染管线冒烟测试：从 index.html 抽取纯函数，验证切分 + texify + KaTeX 可渲染
const fs = require('fs');
const path = require('path');
const html = fs.readFileSync(path.join(__dirname, '..', 'web', 'index.html'), 'utf8');

// 抽取 UNI_MAP 到 formatAI 之间的纯函数段
const start = html.indexOf('const UNI_MAP');
const end = html.indexOf('function renderKatex');
if (start < 0 || end < 0) { console.error('FAIL: 未找到函数段'); process.exit(1); }
let code = html.slice(start, end);
// 提供HAS_LATEX（在函数段中已包含）——eval 执行
eval(code);

const katex = require(path.join(__dirname, '..', 'web', 'katex', 'katex.min.js'));

const cases = [
  '当x=(1+√(1994))/(2)时，多项式(4x^3-1997x-1994)^2001的值为．',
  '若关于x 的一元二次方程kx^2-x+1=0有实数根，则k的取值范围是．',
  '已知△ABC∽△DEF，相似比为 4:1，若 AB=4 且 AB 与 DE 是对应边，则 DE 的长为（\u3000）',
  '设a=√(7)-1，则代数式3a^3+12a^2-6a-12的值为．',
  '已知 x、y 满足方程组 x+y=26，x−y=12，则 x 的值为',
  '◆ 定义\n一次函数 y=kx+b（k≠0）中，当 k>0 时 y 随 x 的增大而增大',
  '斜率 $k = \\frac{y_2-y_1}{x_2-x_1}$，表示倾斜程度。',
  '判别式 b^2-4ac：当 Δ≥0 时方程有实数根，Δ<0 时无实数根。',
  '定期阅读 LangGraph 文档，第 3 章讲了 2021-2022 年的考情。',
  '解：设甲的速度为 x 千米/小时。\n由题意得 6x=3(x+4)。',
];

let bad = 0;
// 模拟浏览器：innerHTML 中的实体在 DOM textNode 里已解码，KaTeX 看到的是解码后文本
const unesc = s => s.replace(/&lt;/g,'<').replace(/&gt;/g,'>').replace(/&amp;/g,'&');
for (const c of cases) {
  const out = formatAI(c);
  // 抽出所有 \( \) 片段与 $...$ 片段，用 KaTeX 实渲染验证
  const texs = [];
  for (const m of out.matchAll(/\\\((.*?)\\\)/gs)) texs.push(unesc(m[1]));
  for (const m of c.matchAll(/\$([^$\n]+)\$/g)) texs.push(m[1]);
  for (const t of texs) {
    try { katex.renderToString(t, { throwOnError: true }); }
    catch (e) { bad++; console.error('KaTeX 渲染失败:', JSON.stringify(t), '←', e.message.slice(0, 80)); }
  }
  console.log('---');
  console.log('IN :', c.replace(/\n/g, '⏎').slice(0, 70));
  console.log('OUT:', out.replace(/\n/g, '⏎').slice(0, 160));
}
console.log(bad === 0 ? '\nALL PASS: 所有片段 KaTeX 渲染成功' : `\nFAIL: ${bad} 个片段渲染失败`);
process.exit(bad === 0 ? 0 : 1);
