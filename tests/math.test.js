const test=require('node:test'),assert=require('node:assert/strict'),m=require('../docs/assets/lab-math.js');
const near=(a,b)=>assert.ok(Math.abs(a-b)<1e-9,`${a} != ${b}`);
test('exact gradient update and loss',()=>{near(m.loss(-2),25);near(m.gradientStep(-2,.2),0);near(m.loss(0),9);near(m.gradientStep(3,1.1),3);});
test('rate cases converge, oscillate, or diverge as advertised',()=>{for(const r of [.05,.2,.8]){let w=-2;for(let i=0;i<100;i++)w=m.gradientStep(w,r);assert.ok(m.loss(w)<25);}assert.ok(m.gradientStep(-2,.8)>3);assert.ok(m.loss(m.gradientStep(-2,1.1))>25);});
test('memory inventory sums and units',()=>{for(const mode of ['fp32','mixed']){const x=m.memory(1000,mode);near(x.total,16e9/2**30);near(x.total,x.weights+x.gradients+x.master+x.optimizer);near(x.decimalGB,16);}near(m.memory(1000,'fp32').master,0);near(m.memory(1000,'mixed').master,4e9/2**30);});
test('memory input limits reject nonfinite and out-of-range',()=>{for(const x of [NaN,Infinity,-1,0,100001])assert.throws(()=>m.memory(x,'fp32'));for(const x of [1,100000])assert.ok(m.memory(x,'fp32').total>0);});
test('causal mask counts and boundaries',()=>{for(let q=0;q<6;q++){assert.equal(Array.from({length:6},(_,k)=>m.allowed(q,k)).filter(Boolean).length,q+1);assert.ok(m.allowed(q,q));}assert.ok(!m.allowed(0,5));assert.ok(m.allowed(5,0));});
