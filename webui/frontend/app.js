/* webui 前端逻辑：上传预览 → POST /api/predict → 图表与文本栏渲染。
 * 只依赖后端契约（webui/backend/schemas.py 的 PredictResponse）与本机
 * vendor/chart.umd.js，无任何外部网络请求。 */
"use strict";

(function () {
  const imageInput = document.getElementById("image-input");
  const predictBtn = document.getElementById("predict-btn");
  const previewWrap = document.getElementById("preview-wrap");
  const previewImg = document.getElementById("preview");
  const statusLine = document.getElementById("status-line");
  const resultCard = document.getElementById("result-card");
  const tModeValue = document.getElementById("t-mode-value");
  const tMeanValue = document.getElementById("t-mean-value");
  const mappingIssues = document.getElementById("mapping-issues");
  const metaLine = document.getElementById("meta-line");
  const preprocessingId = document.getElementById("preprocessing-id");

  let objectUrl = null;
  let chart = null;

  imageInput.addEventListener("change", function () {
    const file = imageInput.files && imageInput.files[0];
    if (!file) return;
    if (objectUrl) URL.revokeObjectURL(objectUrl);
    objectUrl = URL.createObjectURL(file);
    previewImg.src = objectUrl;
    previewWrap.hidden = false;
    predictBtn.disabled = false;
    resultCard.hidden = true;
    statusLine.textContent = "";
  });

  predictBtn.addEventListener("click", async function () {
    const file = imageInput.files && imageInput.files[0];
    if (!file) return;
    predictBtn.disabled = true;
    predictBtn.textContent = "推理中…";
    statusLine.textContent = "";
    try {
      const form = new FormData();
      form.append("image", file);
      const resp = await fetch("/api/predict", { method: "POST", body: form });
      if (!resp.ok) {
        let detail = "HTTP " + resp.status;
        try {
          const err = await resp.json();
          if (err && err.detail) detail = err.detail;
        } catch (_) {
          /* 非 JSON 错误体，保留状态码信息 */
        }
        throw new Error(detail);
      }
      renderResult(await resp.json());
    } catch (err) {
      statusLine.textContent = "推理失败：" + err.message;
    } finally {
      predictBtn.disabled = false;
      predictBtn.textContent = "开始推理";
    }
  });

  function renderResult(data) {
    resultCard.hidden = false;
    tModeValue.textContent = "T " + data.t_mode.toFixed(1);
    tMeanValue.textContent = data.t_mean.toFixed(2);
    preprocessingId.textContent = data.preprocessing_id;
    metaLine.textContent =
      data.model_id + " · SHA-256 " + data.model_sha256.slice(0, 12) + "… · " +
      "输入 " + data.input_id + " · " + data.run.device.toUpperCase() +
      " · torch " + data.run.torch + " · ultralytics " + data.run.ultralytics;
    if (data.class_mapping_issues && data.class_mapping_issues.length) {
      mappingIssues.textContent = "类别表提示：" + data.class_mapping_issues.join("；");
      mappingIssues.hidden = false;
    } else {
      mappingIssues.hidden = true;
    }
    const labels = data.class_labels.t_values.map(function (t) {
      return "T" + t.toFixed(1);
    });
    renderChart(labels, data.probabilities);
  }

  function renderChart(labels, probs) {
    if (chart) chart.destroy();
    chart = new Chart(document.getElementById("prob-chart"), {
      type: "bar",
      data: {
        labels: labels,
        datasets: [
          {
            label: "类别概率",
            data: probs,
            backgroundColor: "rgba(47, 108, 163, 0.55)",
            borderColor: "rgba(24, 74, 118, 1)",
            borderWidth: 1,
          },
          {
            label: "概率分布折线",
            type: "line",
            data: probs,
            borderColor: "rgba(201, 69, 65, 1)",
            backgroundColor: "transparent",
            pointRadius: 3,
            pointHoverRadius: 5,
            tension: 0.25,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        interaction: { mode: "index", intersect: false },
        scales: {
          x: { title: { display: true, text: "Dvorak T-number（按权重内实际类别表）" } },
          y: {
            beginAtZero: true,
            suggestedMax: 1.0,
            title: { display: true, text: "概率（Softmax，未校准）" },
            ticks: { callback: function (v) { return Number(v).toFixed(2); } },
          },
        },
        plugins: {
          legend: { position: "bottom" },
          tooltip: {
            callbacks: {
              label: function (ctx) {
                return ctx.dataset.label + "：" + ctx.parsed.y.toFixed(4);
              },
            },
          },
        },
      },
    });
  }
})();
