// 前端检测：加载页面、收集控制台错误、验证 KaTeX 渲染、切换学习模式、截图（16 项）
// 运行方式：服务须已启动（127.0.0.1:8787）；
//   依赖 playwright-core（本机 Edge 免下 Chromium）：
//   cd .workbuddy/tmp/pwtest && npm i playwright-core && node ../../../scripts/frontend_check.js
//   （或 NODE_PATH=<playwright-core 所在 node_modules> node scripts/frontend_check.js）
const { chromium } = require('playwright-core');

(async () => {
  const results = [];
  const ok = (name, pass, detail='') => {
    results.push({ name, pass, detail });
    console.log(`${pass ? 'PASS' : 'FAIL'}  ${name}${detail ? '  | ' + detail : ''}`);
  };

  const browser = await chromium.launch({
    executablePath: 'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe',
    headless: true,
  });
  const page = await browser.newPage({ viewport: { width: 1280, height: 860 } });
  const consoleErrors = [];
  page.on('console', m => { if (m.type() === 'error') consoleErrors.push(m.text()); });
  page.on('pageerror', e => consoleErrors.push('pageerror: ' + e.message));
  page.on('requestfailed', r => {
    const u = r.url();
    if (!u.includes('bigmodel')) consoleErrors.push('reqfail: ' + u + ' ' + (r.failure()||{}).errorText);
  });

  // 1. 页面加载
  await page.goto('http://127.0.0.1:8787/', { waitUntil: 'load', timeout: 30000 });
  ok('页面加载', true, await page.title());

  // 2. KaTeX 库加载
  await page.waitForTimeout(800);
  const katexReady = await page.evaluate(() => ({
    katex: typeof window.katex !== 'undefined',
    autoRender: typeof window.renderMathInElement === 'function',
  }));
  ok('KaTeX 库加载', katexReady.katex, 'window.katex=' + katexReady.katex);
  ok('auto-render 扩展', katexReady.autoRender, '');

  // 3. 欢迎消息 + 头像
  const welcome = await page.evaluate(() => ({
    bubbles: document.querySelectorAll('.msg.ai').length,
    avatars: document.querySelectorAll('.avatar').length,
  }));
  ok('欢迎消息渲染', welcome.bubbles >= 1, `AI气泡=${welcome.bubbles} 头像=${welcome.avatars}`);

  // 4. 运行状态面板（服务连通）
  await page.waitForTimeout(500);
  const stats = await page.textContent('#stats');
  ok('运行状态面板', !stats.includes('服务未启动'), stats.replace(/\s+/g,' ').slice(0, 60));
  const mode = await page.textContent('#mode');
  ok('模式徽章', mode.includes('AI 老师在线') || mode.includes('基础模式'), mode);

  // 5. 在真实 DOM 中验证数学渲染管线
  const mathTest = await page.evaluate(() => {
    const sample = '当x=(1+√(1994))/(2)时，多项式(4x^3-1997x-1994)^2001的值为．已知△ABC∽△DEF，kx^2-x+1=0';
    const b = document.querySelector('.msg.ai');
    b.innerHTML = formatAI(sample);
    renderKatex(b);
    const vis = b.querySelector('.katex-html') ? b.querySelector('.katex-html').textContent : b.textContent;
    return {
      katexNodes: b.querySelectorAll('.katex').length,
      frac: b.innerHTML.includes('mfrac'),
      sqrt: b.innerHTML.includes('sqrt'),
      rawCaret: vis.includes('^'),      // 只查可见层；MathML annotation 里的 TeX 源码不算残留
      rawSqrt: vis.includes('√'),
    };
  });
  ok('公式节点生成', mathTest.katexNodes > 0, `katex节点=${mathTest.katexNodes}`);
  ok('分数渲染', mathTest.frac, '');
  ok('根号渲染', mathTest.sqrt, '');
  ok('无残留 ^ 记法', !mathTest.rawCaret, '');
  ok('无残留 √ 字符', !mathTest.rawSqrt, '');

  // 6. 学习模式切换
  await page.click('#tab-learn');
  await page.waitForTimeout(1200);
  const learn = await page.evaluate(() => ({
    panelVisible: document.getElementById('learn-panel').style.display !== 'none',
    topicOptions: document.querySelectorAll('#learn-topic option').length,
    pointOptions: document.querySelectorAll('#learn-point option').length,
    bodyClass: document.body.classList.contains('learn'),
  }));
  ok('学习面板显示', learn.panelVisible, '');
  ok('主题色切换', learn.bodyClass, '');
  ok('知识点下拉加载', learn.pointOptions > 0, `板块=${learn.topicOptions} 知识点=${learn.pointOptions}`);
  await page.screenshot({ path: 'D:/游戏库/智能助教/.workbuddy/tmp/pwtest/learn-mode.png' });

  // 7. 刷题模式 + 快捷指令点击发送（走真实 LLM）
  // 切回刷题页签会自动发一条「退出学习」，必须等它完成（发送按钮恢复）再点快捷指令，
  // 否则同用户并发互斥 → 429 busy，后一条消息拿不到真实回复
  await page.click('#tab-quiz');
  await page.waitForFunction(() => !document.getElementById('send').disabled, { timeout: 90000 });
  await page.click('.chip:has-text("一次函数的斜率是什么")');
  await page.waitForSelector('.msg.ai:not(.typing)', { timeout: 90000 });
  // 等流式结束（发送按钮恢复可用）
  await page.waitForFunction(() => !document.getElementById('send').disabled, { timeout: 90000 });
  const msgs = await page.evaluate(() => document.querySelectorAll('.msg.ai').length);
  const lastText = await page.evaluate(() => {
    const all = document.querySelectorAll('.msg.ai');
    return all[all.length - 1].textContent.slice(0, 40);
  });
  ok('对话流程（点击快捷指令→流式回复）', msgs >= 2, `气泡=${msgs} 回复="${lastText}..."`);
  await page.screenshot({ path: 'D:/游戏库/智能助教/.workbuddy/tmp/pwtest/quiz-mode.png', fullPage: false });

  // 8. 控制台错误
  ok('无控制台错误', consoleErrors.length === 0, consoleErrors.slice(0, 3).join(' ; ').slice(0, 150));

  await browser.close();
  const failed = results.filter(r => !r.pass).length;
  console.log(`\n${results.length - failed}/${results.length} 项通过`);
  process.exit(failed ? 1 : 0);
})().catch(e => { console.error('SCRIPT ERROR:', e.message, '\n', (e.stack||'').split('\n').slice(1,4).join('\n')); process.exit(2); });
