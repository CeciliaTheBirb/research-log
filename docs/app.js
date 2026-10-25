const video = document.querySelector('#comparison-video');
const scenes = [...document.querySelectorAll('[data-time]')];
const originalVideoURL = video.querySelector('source').src;
let seekableVideo;
// Some static hosts stream MP4 without byte ranges. A complete in-memory copy
// gives the media element a seekable resource while keeping the download link.
function prepareSeeking() {
  if (!seekableVideo) seekableVideo = (async () => {
    const response = await fetch(originalVideoURL);
    if (!response.ok) throw new Error('Video download failed');
    const url = URL.createObjectURL(await response.blob());
    await new Promise((resolve, reject) => {
      video.addEventListener('loadedmetadata', resolve, { once: true });
      video.addEventListener('error', reject, { once: true });
      video.src = url;
      video.load();
    });
  })().catch(error => { seekableVideo = null; throw error; });
  return seekableVideo;
}
scenes.forEach(scene => scene.addEventListener('click', async () => {
  scenes.forEach(button => { button.disabled = true; button.setAttribute('aria-busy', 'true'); });
  try {
    await prepareSeeking();
    video.currentTime = Number(scene.dataset.time);
    await video.play();
  } catch {
    video.controls = true;
  } finally {
    scenes.forEach(button => { button.disabled = false; button.removeAttribute('aria-busy'); });
  }
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
