def f():
 with open('new.py', encoding='utf-8') as f: lines = f.readlines(); [print(f'{i+1}: {l.strip()}') for i, l in enumerate(lines) if 'is_braking' in l]
f()