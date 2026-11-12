const {chromium}=require(process.env.PLAYWRIGHT_MODULE || 'C:/Users/yfmo/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const assert=require('node:assert/strict');
const path=require('node:path');
(async()=>{
 const browser=await chromium.launch({headless:true,channel:process.env.PLAYWRIGHT_CHANNEL||'msedge'});
 try{
  const page=await browser.newPage({viewport:{width:1440,height:1050}});
  await page.addInitScript(()=>{Object.defineProperty(crypto,'randomUUID',{value:undefined});});
  const errors=[];page.on('pageerror',e=>errors.push(e.message));
  await page.goto('http://127.0.0.1:18090/intergrated_test.html#run');
  await page.locator('#episodeFailure').click();
  await page.waitForFunction(()=>document.querySelectorAll('#reviewRounds button').length===2);
  await page.waitForFunction(()=>!document.querySelector('#reviewSave').disabled);
  const state=async()=>await(await page.request.get('http://127.0.0.1:18090/api/test-state')).json();
  let d=await state();assert.equal(d.collection_state,'stopped');assert.equal(d.result,null);
  await page.locator('#reviewSave').click();
  await page.waitForFunction(()=>document.querySelector('#reviewMessage').textContent.includes('请补齐'));
  assert.equal((await state()).result,null);
  await page.locator('#reviewRoundNote').fill('第一次夹爪偏高，人工重新对准。');
  await page.locator('#reviewEditor input[value=grasp_failed]').check();
  await page.locator('#reviewRounds button').nth(1).click();
  await page.locator('#reviewRoundNote').fill('第二次门把手处停滞，调整开门方向。');
  await page.locator('#reviewRounds button').nth(0).click();
  assert.match(await page.locator('#reviewRoundNote').inputValue(),/第一次/);
  await page.waitForTimeout(800);
  d=await state();assert.equal(d.result,null);assert.deepEqual(d.annotations,{});
  await page.screenshot({path:path.resolve(__dirname,'../runtime/failure-review-desktop.png'),fullPage:true});
  await page.setViewportSize({width:390,height:844});
  await page.screenshot({path:path.resolve(__dirname,'../runtime/failure-review-mobile.png'),fullPage:true});
  await page.locator('#reviewSave').click();
  await page.waitForFunction(()=>!document.querySelector('#episodeDialog').open);
  d=await state();assert.equal(d.archive_state,'complete');assert.equal(d.result.success,false);
  assert.equal(Object.keys(d.annotations).length,2);assert.equal(d.stop_count,1);
  await page.locator('#dataTab').click();
  await page.waitForFunction(()=>document.querySelector('#recordCount').textContent.startsWith('1 条'));
  assert.deepEqual(errors,[]);
  console.log('INLINE_BROWSER_OK',JSON.stringify({rounds:2,noAutosave:true,archived:true,errors}));
 }finally{await browser.close();}
})().catch(e=>{console.error(e);process.exit(1)});
