"""Inject consent-aware GA4 into Streamlit's outer HTML shell at startup."""

from __future__ import annotations

import json
import re
from pathlib import Path


START_MARKER = "<!-- NFL_ANALYTICS_GA4_START -->"
END_MARKER = "<!-- NFL_ANALYTICS_GA4_END -->"
MEASUREMENT_ID_PATTERN = re.compile(r"^G-[A-Z0-9]+$")


def build_ga4_shell(measurement_id: str) -> str:
    """Return a basic-consent GA4 shell for the outer Streamlit document."""

    measurement_id = measurement_id.strip().upper()
    if not MEASUREMENT_ID_PATTERN.fullmatch(measurement_id):
        raise ValueError("Invalid GA4 Measurement ID.")
    ga_id = json.dumps(measurement_id)
    return f"""{START_MARKER}
<style>
#nap-consent{{position:fixed;left:50%;bottom:24px;transform:translateX(-50%);z-index:2147483646;width:min(760px,calc(100vw - 32px));padding:18px 20px;border:1px solid #315170;border-radius:12px;background:#0b1b2e;color:#f4f7fb;box-shadow:0 12px 40px rgba(0,0,0,.45);font:14px/1.45 system-ui,sans-serif}}
#nap-consent strong{{display:block;font-size:16px;margin-bottom:6px}}#nap-consent p{{margin:0 0 14px;color:#c7d5e8}}
#nap-consent-actions{{display:flex;gap:10px;justify-content:flex-end;flex-wrap:wrap}}#nap-consent button{{border:1px solid #42617e;border-radius:8px;padding:9px 14px;cursor:pointer;font-weight:700}}
#nap-consent-essential{{background:#14253c;color:#e8eef7}}#nap-consent-accept{{background:#36e39a;color:#06111f;border-color:#36e39a!important}}
#nap-privacy{{position:fixed;left:10px;bottom:8px;z-index:2147483645;border:0;background:transparent;color:#8ea5c4;font:11px system-ui,sans-serif;cursor:pointer}}
</style>
<script>
(function(){{
  const GA_ID={ga_id}, CONSENT_KEY='nap_analytics_consent';
  let loaded=false,lastPage='';
  function lang(){{return new URL(location.href).searchParams.get('language')==='HU'?'HU':'EN';}}
  function page(){{return new URL(location.href).searchParams.get('page')||'OVERVIEW';}}
  function copy(){{return lang()==='HU'?{{title:'Anonim látogatottsági statisztika',body:'Az oldal anonim látogatottsági adatokat mérhet a Google Analytics segítségével. Az adatok az oldal fejlesztését segítik; hirdetési és személyre szabási funkciókat nem használunk.',accept:'Statisztikai sütik engedélyezése',essential:'Csak szükséges',privacy:'Adatvédelem'}}:{{title:'Anonymous usage analytics',body:'This site can use Google Analytics to measure anonymous visits and improve the platform. Advertising and personalization features are disabled.',accept:'Allow analytics cookies',essential:'Essential only',privacy:'Privacy'}};}}
  function track(){{if(!loaded)return;const p=page(),key=p+':'+lang();if(key===lastPage)return;lastPage=key;gtag('event','page_view',{{page_title:p,page_location:location.href,page_path:'/'+p.toLowerCase(),dashboard_page:p,dashboard_language:lang()}});}}
  function load(){{if(loaded)return;loaded=true;window.dataLayer=window.dataLayer||[];window.gtag=function(){{dataLayer.push(arguments)}};gtag('js',new Date());gtag('config',GA_ID,{{send_page_view:false,allow_google_signals:false,allow_ad_personalization_signals:false}});const s=document.createElement('script');s.async=true;s.src='https://www.googletagmanager.com/gtag/js?id='+encodeURIComponent(GA_ID);document.head.appendChild(s);track();}}
  function hide(){{document.getElementById('nap-consent')?.remove();}}
  function choose(value){{localStorage.setItem(CONSENT_KEY,value);hide();if(value==='granted')load();}}
  function banner(){{hide();const c=copy(),box=document.createElement('div');box.id='nap-consent';box.innerHTML='<strong>'+c.title+'</strong><p>'+c.body+'</p><div id="nap-consent-actions"><button id="nap-consent-essential">'+c.essential+'</button><button id="nap-consent-accept">'+c.accept+'</button></div>';document.body.appendChild(box);document.getElementById('nap-consent-essential').onclick=()=>choose('denied');document.getElementById('nap-consent-accept').onclick=()=>choose('granted');}}
  function init(){{const c=copy(),privacy=document.createElement('button');privacy.id='nap-privacy';privacy.textContent=c.privacy;privacy.onclick=()=>{{localStorage.removeItem(CONSENT_KEY);banner();}};document.body.appendChild(privacy);const state=localStorage.getItem(CONSENT_KEY);if(state==='granted')load();else if(state!=='denied')banner();setInterval(track,750);}}
  if(document.readyState==='loading')document.addEventListener('DOMContentLoaded',init);else init();
}})();
</script>
{END_MARKER}"""


def inject_ga4_into_index(index_file: Path, measurement_id: str) -> bool:
    """Insert or replace the managed GA4 block in a Streamlit index file."""

    content = index_file.read_text(encoding="utf-8")
    block = build_ga4_shell(measurement_id)
    pattern = re.compile(
        re.escape(START_MARKER) + r".*?" + re.escape(END_MARKER),
        flags=re.DOTALL,
    )
    if pattern.search(content):
        updated = pattern.sub(block, content)
    else:
        if "</head>" not in content:
            raise ValueError("Streamlit index does not contain </head>.")
        updated = content.replace("</head>", f"{block}\n</head>", 1)
    if updated == content:
        return False
    index_file.write_text(updated, encoding="utf-8")
    return True
