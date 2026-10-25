const video = document.querySelector('#comparison-video');
const scenes = [...document.querySelectorAll('[data-time]')];
scenes.forEach(scene => scene.addEventListener('click', () => {
  const seek = () => { video.currentTime = Number(scene.dataset.time); video.play().catch(() => {}); };
  if (video.readyState >= 1) seek();
  else { video.addEventListener('loadedmetadata', seek, { once: true }); video.load(); }
}));
video.addEventListener('timeupdate', () => {
  const i = video.currentTime < 6.05 ? 0 : video.currentTime < 12.1 ? 1 : 2;
  scenes.forEach((scene, k) => scene.classList.toggle('active', i === k));
});
const policyButtons = [...document.querySelectorAll('[data-policy]')];
function policy(name) {
  const noisy = name === 'noisy';
  policyButtons.forEach(button => { const active = button.dataset.policy === name; button.classList.toggle('active', active); button.setAttribute('aria-pressed', String(active)); });
  const track = document.querySelector('#block-track'); track.replaceChildren();
  for (let i = 0; i <= 10; i++) {
    const block = document.createElement('div');
    const reuse = noisy && i >= 2;
    block.className = `block${i === 0 ? ' anchor' : ''}${reuse ? ' noise' : ''}`;
    for (const [tag, text] of [['span', i === 0 ? '首帧' : `块 ${i}`], ['b', i === 0 ? 'A' : i === 1 ? '4' : '2'], ['small', reuse ? 'NOISY KV' : 'CLEAN KV']]) {
      const e = document.createElement(tag); e.textContent = text; block.append(e);
    }
    track.append(block);
  }
  document.querySelector('#policy-description').textContent = noisy ? '后续 9 个块复用 σ=0.625 最后一步输入的 KV，省去各自的 clean commit。' : '每个块完成 denoise 后，额外执行一次干净历史的 KV 写入。';
  document.querySelector('#call-count').textContent = noisy ? '24 次模型调用' : '33 次模型调用';
}
policyButtons.forEach(button => button.addEventListener('click', () => policy(button.dataset.policy)));
policy('clean');
