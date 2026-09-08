import random,time

def conf(b):
    n=len(b)
    c=0
    for i in range(n):
        for j in range(i+1,n):
            if b[i]==b[j] or abs(b[i]-b[j])==abs(i-j):
                c+=1
    return c

def hillclimb(n,maxr=200):
    t0=time.perf_counter()
    steps=0
    for rs in range(1,maxr+1):
        b=[random.randint(0,n-1) for _ in range(n)]
        while True:
            cur=conf(b)
            if cur==0:
                return b,rs,steps,time.perf_counter()-t0
            bm=None
            bv=cur
            for col in range(n):
                og=b[col]
                for row in range(n):
                    if row==og:
                        continue
                    b[col]=row
                    v=conf(b)
                    if v<bv:
                        bv=v
                        bm=(col,row)
                    b[col]=og
            steps+=1
            if bm is None:
                break
            b[bm[0]]=bm[1]
    return None,maxr,steps,time.perf_counter()-t0

def backtrack(n):
    t0=time.perf_counter()
    b=[-1]*n
    nd=[0]
    def safe(row,col):
        for r in range(row):
            c=b[r]
            if c==col or abs(c-col)==abs(r-row):
                return False
        return True
    def bt(row):
        if row==n:
            return True
        for col in range(n):
            nd[0]+=1
            if safe(row,col):
                b[row]=col
                if bt(row+1):
                    return True
                b[row]=-1
        return False
    ok=bt(0)
    return (b if ok else None),nd[0],time.perf_counter()-t0

def show(b):
    n=len(b)
    for r in range(n):
        print(" ".join("Q" if b[c]==r else "." for c in range(n)))

def menu():
    print("N-QUEENS SEARCH")
    s=input("enter N  ").strip()
    n=int(s) if s else 8
    print("1.HILL CLIMBING(local search) 2.BACKTRACKING(CSP) 3.RUN BOTH")
    ch=int(input("choice: ") or 3)
    if ch in(1,3):
        b,rs,st,t=hillclimb(n)
        print(f"HILLCLIMB restarts={rs} steps={st} t={t:.5f}s solved={b is not None}")
        if b:
            show(b)
    if ch in(2,3):
        b,nd,t=backtrack(n)
        print(f"BACKTRACK nodes={nd} t={t:.5f}s solved={b is not None}")
        if b:
            show(b)

if __name__=="__main__":
    menu()