const $=id=>document.getElementById(id);
const DEFAULT="http://127.0.0.1:8000";
chrome.storage.sync.get({apiBase:DEFAULT},v=>{$("api").value=v.apiBase||DEFAULT});
function status(message,kind){const n=$("status");n.textContent=message;n.className="status "+(kind||"")}
$("save").addEventListener("click",()=>{let base=$("api").value.trim().replace(/\/$/,"");if(!/^https?:\/\//i.test(base)){status("Enter a full address starting with https:// or http://.","bad");return}chrome.storage.sync.set({apiBase:base},()=>status("Saved. The extension will use this API address.","ok"))});
$("test").addEventListener("click",()=>{status("Checking API…");chrome.runtime.sendMessage({type:"HG_TEST_API"},r=>{if(chrome.runtime.lastError||!r?.ok){status(r?.error||chrome.runtime.lastError?.message||"Could not connect. Check the address and deployment.","bad");return}status("Connected · "+(r.health.version||"HealthcareGuard API")+" · model ready: "+r.health.model_loaded,"ok")})});
$("open").addEventListener("click",()=>chrome.runtime.sendMessage({type:"HG_OPEN_DASHBOARD"}));
