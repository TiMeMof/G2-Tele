const {chromium}=require(process.env.PLAYWRIGHT_MODULE || 'C:/Users/yfmo/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const assert=require('node:assert/strict');
const path=require('node:path');
(async()=>{
 const browser=await chromium.launch({headless:true,channel:process.env.PLAYWRIGHT_CHANNEL||'msedge'});
 try{
  const page=await browser.newPage({viewport:{width:1440,height:1050}});
  const errors=[];page.on('pageerror',e=>errors.push(e.message));
  await page.goto('http://127.0.0.1:18089/intergrated_test.html#data');
  await page.getByRole('button',{name:'采集数据',exact:true}).click().catch(()=>{});
  await page.waitForFunction(()=>document.querySelectorAll('#phaseTrack .human').length===3);
  assert.equal(await page.locator('#finalizeSession').isDisabled(),true);
  await page.locator('#phaseTrack .human').nth(0).click();
  await page.locator('#annotationReasons input[value=grasp_failed]').check();
  await page.locator('#annotationNote').fill('第一次：夹爪偏高，重新对准');
  await page.locator('#saveRoundAnnotation').click();
  await page.waitForFunction(()=>document.querySelector('#annotationSaveStatus')?.textContent==='已保存');
  await page.locator('#nextRound').click();
  await page.locator('#annotationNote').fill('第二次：门把手附近停滞');
  // Switch before debounce fires; the annotation must remain attached to round two.
  await page.locator('#nextRound').click();
  await page.locator('#annotationReasons input[value=wrong_pose]').check();
  await page.locator('#annotationNote').fill('第三次：调整目标位置');
  await page.locator('#saveRoundAnnotation').click();
  await page.waitForFunction(()=>document.querySelector('#finalizeSession')?.disabled===false);
  await page.locator('#previousRound').click();
  assert.equal(await page.locator('#annotationNote').inputValue(),'第二次：门把手附近停滞');
  await page.locator('#finalizeSession').click();
  await page.waitForFunction(()=>document.querySelector('#annotationProgress')?.textContent==='已完成归档');
  await page.reload();
  await page.waitForFunction(()=>document.querySelector('#annotationProgress')?.textContent==='已完成归档');
  assert.deepEqual(errors,[]);
  await page.screenshot({path:path.resolve(__dirname,'../runtime/review-desktop.png'),fullPage:true});
  await page.setViewportSize({width:390,height:844});
  await page.screenshot({path:path.resolve(__dirname,'../runtime/review-mobile.png'),fullPage:true});
  const records=await (await page.request.get('http://127.0.0.1:18089/api/records')).json();
  const detail=await (await page.request.get('http://127.0.0.1:18089/api/records/'+records.records[0].episode_id)).json();
  assert.equal(detail.episode.archive_state,'complete');
  assert.equal(detail.episode.rounds[1].annotation.note,'第二次：门把手附近停滞');
  console.log('BROWSER_REVIEW_OK',JSON.stringify({rounds:detail.episode.rounds.length,archived:true,errors}));
 } finally{await browser.close();}
})().catch(e=>{console.error(e);process.exit(1)});
