#!/usr/bin/env python3
"""Create a five-league PickFoot CSV from OpenFootball public-domain JSON.
Run on a machine with Internet access: python build_pickfoot_2025_26.py
"""
import csv
import json
import sys
import urllib.request
from collections import Counter
from datetime import date
from pathlib import Path

BASE = 'https://raw.githubusercontent.com/openfootball/football.json/master/2025-26/'
LEAGUES = {'PL': ('en.1.json',380), 'PD': ('es.1.json',380), 'BL1': ('de.1.json',306), 'SA': ('it.1.json',380), 'FL1': ('fr.1.json',306)}
FIELDS = ['league','season','match_date','home_team','away_team','home_goals','away_goals','result']

def extract_score(raw):
    if isinstance(raw, dict):
        raw = raw.get('ft')
    if not isinstance(raw, list) or len(raw) != 2 or any(type(x) is not int or x < 0 for x in raw):
        return None
    return raw

def main():
    rows=[]
    for league,(filename,expected) in LEAGUES.items():
        url=BASE+filename
        print(f'Fetching {league}: {url}', flush=True)
        try:
            req=urllib.request.Request(url, headers={'User-Agent':'PickFoot-Manual-Data-Preparation/1.0'})
            with urllib.request.urlopen(req,timeout=30) as response:
                payload=json.load(response)
        except Exception as exc:
            sys.exit(f'ERROR: failed to fetch {league}: {exc}; no combined CSV was written')
        seen=set()
        for match in payload.get('matches',[]):
            score=extract_score(match.get('score'))
            if score is None:
                continue
            day=match.get('date')
            home=match.get('team1')
            away=match.get('team2')
            try:
                date.fromisoformat(day)
            except (ValueError,TypeError):
                sys.exit(f'ERROR: invalid date for {league}: {day!r}')
            if not home or not away or home == away:
                sys.exit(f'ERROR: invalid teams in {league}: {home!r} / {away!r}')
            key=(day,home,away)
            if key in seen:
                sys.exit(f'ERROR: duplicate match in {league}: {key}')
            seen.add(key)
            hg,ag=score
            rows.append(dict(league=league,season='2025/26',match_date=day,home_team=home,away_team=away,home_goals=hg,away_goals=ag,result='H' if hg>ag else 'A' if hg<ag else 'D'))
        count=len(seen)
        print(f'  {count}/{expected} scored matches')
        if count != expected:
            sys.exit(f'ERROR: {league} has {count} completed matches, expected {expected}. Check source before using. No combined CSV was written.')
    rows.sort(key=lambda x:(x['match_date'],x['league'],x['home_team'],x['away_team']))
    target=Path(__file__).resolve().parent/'pickfoot_2025_26_top5_matches.csv'
    with target.open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=FIELDS)
        writer.writeheader();writer.writerows(rows)
    print(f'Created {target} with {len(rows)} matches')
    print('Counts:',dict(Counter(row['league'] for row in rows)))
    print('Source: OpenFootball football.json, public domain; verify local PickFoot league codes before training.')

if __name__=='__main__':
    main()
