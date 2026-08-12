---
comments: true
statistics: true
---

<link href="https://fonts.googleapis.com/css2?family=Noto+Serif+SC:wght@400;600;700;900&display=swap" rel="stylesheet">

<div class="newspaper">

  <!-- 报头 -->
  <header class="masthead">
    <div class="dateline">
      <span class="paper-vol">VOL. 01</span>
      <span class="paper-date" id="paper-date">——</span>
    </div>
    <h1 class="paper-title">RICCER'S BLOG</h1>
    <p class="paper-subtitle">里瑟博客</p>
    <p class="paper-motto">记录 AI 学习之路，分享思考与发现</p>
    <div class="masthead-rule"></div>
  </header>

  <!-- 头条：站点统计 -->
  <section class="headline-block">
    <p class="kicker">今日头条 · HEADLINE</p>
    <h2 class="headline">本站共刊登 {{ pages }} 个版面，累计 {{ words }} 字</h2>
    <div class="stats-row">
      <div class="stat-card">
        <span class="stat-number">{{ pages }}</span>
        <span class="stat-label">版面数 · PAGES</span>
      </div>
      <div class="stat-card">
        <span class="stat-number">{{ words }}</span>
        <span class="stat-label">总字数 · WORDS</span>
      </div>
    </div>
  </section>

  <!-- 本期导读 -->
  <section class="guide-block">
    <h3 class="guide-title">本期导读 · CONTENTS</h3>
    <div class="guide-columns">
      <div class="guide-item">
        <span class="guide-name">模型</span>
        <span class="guide-dots"></span>
        <a class="guide-link" href="AI/model/">阅读 →</a>
      </div>
      <div class="guide-item">
        <span class="guide-name">强化学习</span>
        <span class="guide-dots"></span>
        <a class="guide-link" href="AI/rl/">阅读 →</a>
      </div>
      <div class="guide-item">
        <span class="guide-name">AI Infra</span>
        <span class="guide-dots"></span>
        <a class="guide-link" href="AI/infra/">阅读 →</a>
      </div>
      <div class="guide-item">
        <span class="guide-name">推理加速</span>
        <span class="guide-dots"></span>
        <a class="guide-link" href="AI/inference_accelerating/">阅读 →</a>
      </div>
    </div>
    <p class="guide-footer">—— 更多内容请浏览上方导航栏 ——</p>
  </section>

</div>

<script>
  document.addEventListener("DOMContentLoaded", function () {
    var el = document.getElementById("paper-date");
    if (!el) return;
    var now = new Date();
    var week = ["星期日", "星期一", "星期二", "星期三", "星期四", "星期五", "星期六"];
    el.textContent =
      now.getFullYear() + "年" +
      (now.getMonth() + 1) + "月" +
      now.getDate() + "日 " +
      week[now.getDay()];
  });
</script>
