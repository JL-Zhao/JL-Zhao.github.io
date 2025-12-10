---
comments: true
statistics: true
---

<link href="https://fonts.googleapis.com/css2?family=Lobster&display=swap" rel="stylesheet">

<style>
  .welcome-container {
    display: flex;
    justify-content: center;
    overflow: hidden;
  }

  .welcome-title {
    font-size: 12rem;
    font-family: 'Lobster', cursive;
    background: linear-gradient(90deg, #ff0000, #ffff00, #00ff00, #00ffff, #0000ff, #ff00ff, #ff0000, #ff0000, #ffff00, #00ff00);
    background-size: 400%;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: gradientAnimation 10s linear infinite;
  }

  @keyframes gradientAnimation {
    0% {
      background-position: 0% 50%;
    }
    100% {
      background-position: 100% 50%;
    }
  }

  .stats-container {
    display: flex;
    flex-direction: column;
    align-items: center;
    padding: 1rem;
    margin-top: 2rem;
    border-radius: 10px;
    background: linear-gradient(45deg, #f0f8ff, #f5f5f5);
    box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
  }

  .stats-container h2 {
    font-size: 1.3rem;
    color: #333;
    font-family: 'Lobster', cursive;
    margin-bottom: 0.5rem;
    margin-top: 0;
  }

  .stats-container .info {
    font-size: 0.9rem;
    color: #555;
    background-color: rgba(255, 255, 255, 0.8);
    padding: 0.5rem;
    border-radius: 8px;
    text-align: center;
    font-family: '楷体', sans-serif;
  }
</style>


<div class="stats-container">
  <h2>站点统计</h2>
  <div class="info">本站共有 {{ pages }} 个页面，{{ words }} 个字。</div>
</div>