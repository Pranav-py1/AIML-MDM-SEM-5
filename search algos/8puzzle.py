import heapq,time
from collections import deque

GOAL=(1,2,3,4,5,6,7,8,0)

class Node:
    def __init__(s,st,par,g,h):
        s.st=st
        s.par=par
        s.g=g
        s.h=h
    def f(s):
        return s.g+s.h

def nbrs(st):
    i=st.index(0)
    r,c=i//3,i%3
    res=[]
    for dr,dc in[(-1,0),(1,0),(0,-1),(0,1)]:
        nr,nc=r+dr,c+dc
        if 0<=nr<3 and 0<=nc<3:
            j=nr*3+nc
            l=list(st)
            l[i],l[j]=l[j],l[i]
            res.append(tuple(l))
    return res

def man(st):
    h=0
    for i,v in enumerate(st):
        if v==0:
            continue
        gi=GOAL.index(v)
        r1,c1=i//3,i%3
        r2,c2=gi//3,gi%3
        h+=abs(r1-r2)+abs(c1-c2)
    return h

def inv(st):
    a=[x for x in st if x!=0]
    c=0
    for i in range(len(a)):
        for j in range(i+1,len(a)):
            if a[i]>a[j]:
                c+=1
    return c

def solvable(st):
    return inv(st)%2==0

def path(nd):
    p=[]
    while nd:
        p.append(nd.st)
        nd=nd.par
    return p[::-1]

def solve(start,mode):
    t0=time.perf_counter()
    exp=0
    mf=1
    cnt=0
    root=Node(start,None,0,man(start))
    vis={start:0}
    dq=deque([root]) if mode<=2 else None
    pq=[(root.h if mode==4 else root.f(),cnt,root)] if mode>2 else None
    frontier=dq if mode<=2 else pq
    while frontier:
        mf=max(mf,len(frontier))
        if mode==1:
            cur=dq.popleft()
        elif mode==2:
            cur=dq.pop()
        else:
            _,_,cur=heapq.heappop(pq)
        if cur.st==GOAL:
            t=time.perf_counter()-t0
            return path(cur),exp,mf,t
        exp+=1
        for nb in nbrs(cur.st):
            ng=cur.g+1
            if nb not in vis or ng<vis[nb]:
                vis[nb]=ng
                child=Node(nb,cur,ng,man(nb))
                if mode<=2:
                    dq.append(child)
                else:
                    cnt+=1
                    pr=child.g if mode==3 else child.h if mode==4 else child.f()
                    heapq.heappush(pq,(pr,cnt,child))
        frontier=dq if mode<=2 else pq
    return None,exp,mf,time.perf_counter()-t0

NAMES={1:"BFS",2:"DFS",3:"UCS",4:"GREEDY",5:"A*"}

def run(st,mode):
    p,exp,mf,t=solve(st,mode)
    pl=len(p)-1 if p else -1
    print(f"{NAMES[mode]:<8}path={pl:<6}exp={exp:<6}mf={mf:<6}t={t:.5f}s")
    return pl,exp,mf,t

def menu():
    s=input("enter 9 tiles space-separated (blank=use sample): ").strip()
    st=tuple(int(x) for x in s.split()) if s else (4,1,3,2,0,6,7,5,8)
    if not solvable(st):
        print("unsolvable state, using sample instead")
        st=(4,1,3,2,0,6,7,5,8)
    print("start:",st,"goal:",GOAL)
    print("1.BFS 2.DFS 3.UCS 4.GREEDY 5.A* 6.RUN ALL")
    ch=int(input("choice: ") or 6)
    if ch==6:
        for m in range(1,6):
            run(st,m)
    else:
        run(st,ch)

if __name__=="__main__":
    menu()