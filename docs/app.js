// Preserve links shared before the GAN experiment received its own dated page.
if (['#noisy-gan', '#gan-results'].includes(location.hash) && !location.pathname.endsWith('2026-10-27.html')) {
  location.replace(`2026-10-27.html${location.search}${location.hash}`);
}
const video = document.querySelector('#comparison-video');
const scenes = [...document.querySelectorAll('[data-time]')];
const originalVideoURL = video?.querySelector('source')?.src;
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
video?.addEventListener('timeupdate', () => {
  const i = video.currentTime < 6.05 ? 0 : video.currentTime < 12.1 ? 1 : 2;
  scenes.forEach((scene, k) => scene.classList.toggle('active', i === k));
});

const rainVideo = document.querySelector('#rain-video');
const rainButtons = [...document.querySelectorAll('[data-rain]')];
const rainNames = {'2p5mps':'直行 2.5 m/s', '0p5mps':'慢行 0.5 m/s', 'stop':'静止'};
const rainCopies = new Map();
rainButtons.forEach(button => button.addEventListener('click', async () => {
  const tag = button.dataset.rain;
  const source = `assets/rain-comparison-${tag}.mp4`;
  rainButtons.forEach(b => { b.disabled = true; b.setAttribute('aria-busy', 'true'); });
  try {
    if (!rainCopies.has(tag)) {
      const response = await fetch(source);
      if (!response.ok) throw new Error('Video download failed');
      rainCopies.set(tag, URL.createObjectURL(await response.blob()));
    }
    await new Promise((resolve, reject) => {
      const cleanup = () => { rainVideo.removeEventListener('loadedmetadata', loaded); rainVideo.removeEventListener('error', failed); };
      const loaded = () => { cleanup(); resolve(); };
      const failed = () => { cleanup(); reject(new Error('Video load failed')); };
      rainVideo.addEventListener('loadedmetadata', loaded);
      rainVideo.addEventListener('error', failed);
      rainVideo.src = rainCopies.get(tag);
      rainVideo.poster = `assets/rain-poster-${tag}.jpg`;
      rainVideo.load();
    });
    rainButtons.forEach(b => { const selected = b === button; b.classList.toggle('active', selected); b.setAttribute('aria-pressed', String(selected)); });
    document.querySelector('#rain-action-label').textContent = `${rainNames[tag]} · 四模型同步播放`;
    document.querySelector('#rain-download').href = source;
    await rainVideo.play();
  } catch {
    document.querySelector('#rain-action-label').textContent = '视频暂未加载，请重试或下载观看。';
    document.querySelector('#rain-download').href = source;
  } finally {
    rainButtons.forEach(b => { b.disabled = false; b.removeAttribute('aria-busy'); });
  }
}));
const playbackVideos = [...document.querySelectorAll('video')];
playbackVideos.forEach(current => current.addEventListener('play', () => {
  playbackVideos.forEach(other => { if (other !== current) other.pause(); });
}));
const sidebarToggle = document.querySelector('.sidebar-toggle');
sidebarToggle.addEventListener('click', () => {
  if (matchMedia('(max-width: 800px)').matches) {
    const open = document.body.classList.toggle('sidebar-open');
    sidebarToggle.setAttribute('aria-expanded', String(open));
  } else {
    const closed = document.body.classList.toggle('sidebar-collapsed');
    sidebarToggle.setAttribute('aria-expanded', String(!closed));
  }
});
document.querySelectorAll('.entry-nav a, .date-link').forEach(link => link.addEventListener('click', () => {
  if (matchMedia('(max-width: 800px)').matches) {
    document.body.classList.remove('sidebar-open'); sidebarToggle.setAttribute('aria-expanded', 'false');
  }
}));
if (matchMedia('(max-width: 800px)').matches) sidebarToggle.setAttribute('aria-expanded', 'false');
