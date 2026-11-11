const MODEL_URL = "https://huggingface.co/Xenova/segformer-b0-finetuned-cityscapes-512-1024/resolve/498e4add74041c22a9908388f600d836d57f06ea/onnx/model.onnx";
const INPUT_SIZE = 512;
const MEAN = [0.485, 0.456, 0.406];
const STD = [0.229, 0.224, 0.225];
let roadSession;

function canvas(width, height) {
  return Object.assign(document.createElement("canvas"), { width, height });
}

function fillSmallHoles(mask, width, height) {
  const seen = new Uint8Array(mask.length);
  const limit = Math.max(1, Math.floor(mask.length * 0.0015));
  for (let start = 0; start < mask.length; start++) {
    if (mask[start] || seen[start]) continue;
    const component = [start];
    seen[start] = 1;
    let touchesEdge = false;
    for (let i = 0; i < component.length; i++) {
      const pos = component[i], x = pos % width, y = Math.floor(pos / width);
      if (!x || !y || x === width - 1 || y === height - 1) touchesEdge = true;
      for (const next of [x ? pos - 1 : -1, x + 1 < width ? pos + 1 : -1,
                         y ? pos - width : -1, y + 1 < height ? pos + width : -1]) {
        if (next < 0 || mask[next] || seen[next]) continue;
        seen[next] = 1;
        component.push(next);
      }
    }
    if (!touchesEdge && component.length <= limit) {
      for (const pos of component) mask[pos] = 1;
    }
  }
}

async function segmentRoad(bitmap, ort) {
  roadSession ??= ort.InferenceSession.create(MODEL_URL, { executionProviders: ["wasm"] })
    .catch(error => { roadSession = undefined; throw error; });
  const session = await roadSession;
  const input = canvas(INPUT_SIZE, INPUT_SIZE).getContext("2d", { willReadFrequently: true });
  input.drawImage(bitmap, 0, 0, INPUT_SIZE, INPUT_SIZE);
  const { data } = input.getImageData(0, 0, INPUT_SIZE, INPUT_SIZE);
  const area = INPUT_SIZE ** 2;
  const values = new Float32Array(3 * area);
  for (let channel = 0; channel < 3; channel++) {
    for (let i = 0; i < area; i++) {
      values[channel * area + i] = (data[4 * i + channel] / 255 - MEAN[channel]) / STD[channel];
    }
  }
  const outputs = await session.run({
    [session.inputNames[0]]: new ort.Tensor("float32", values, [1, 3, INPUT_SIZE, INPUT_SIZE])
  });
  const { dims: [, classes, height, width], data: logits } = outputs[session.outputNames[0]];
  if (classes !== 19 || !height || !width) throw new Error("도로 분할 모델 출력 형식이 올바르지 않습니다.");
  const pixels = width * height;
  const mask = new Uint8Array(pixels);
  let roadPixels = 0;
  for (let i = 0; i < pixels; i++) {
    let isRoad = true;
    for (let c = 1; c < classes; c++) {
      if (logits[c * pixels + i] > logits[i]) { isRoad = false; break; }
    }
    mask[i] = isRoad ? 1 : 0;
    roadPixels += mask[i];
  }
  if (roadPixels / pixels < 0.02) throw new Error("도로 영역이 충분히 잡히지 않았습니다. 원본 모드로 다시 시도해 주세요.");
  fillSmallHoles(mask, width, height);
  const alpha = canvas(width, height);
  const ctx = alpha.getContext("2d");
  const rgba = ctx.createImageData(width, height);
  for (let i = 0; i < pixels; i++) rgba.data[4 * i + 3] = mask[i] ? 255 : 0;
  ctx.putImageData(rgba, 0, 0);
  return alpha;
}

export async function prepareMaskedImage(blob, mode, polygon, ort) {
  if (mode === "polygon" && polygon.length < 3) throw new Error("사진에서 도로 경계점을 3개 이상 선택해 주세요.");
  const bitmap = await createImageBitmap(blob, { colorSpaceConversion: "none", premultiplyAlpha: "none" });
  try {
    const mask = mode === "road" ? await segmentRoad(bitmap, ort) : null;
    const size = 384;
    const scale = size / Math.max(bitmap.width, bitmap.height);
    const width = Math.max(1, Math.round(bitmap.width * scale));
    const height = Math.max(1, Math.round(bitmap.height * scale));
    const crop = canvas(width, height);
    const ctx = crop.getContext("2d");
    if (!mask) {
      ctx.beginPath();
      polygon.forEach(([x, y], i) => i ? ctx.lineTo(x * width, y * height) : ctx.moveTo(x * width, y * height));
      ctx.closePath();
      ctx.clip();
    }
    ctx.drawImage(bitmap, 0, 0, width, height);
    if (mask) {
      ctx.globalCompositeOperation = "destination-in";
      ctx.imageSmoothingEnabled = false;
      ctx.drawImage(mask, 0, 0, width, height);
    }
    const padded = canvas(size, size);
    const out = padded.getContext("2d");
    out.fillStyle = "#808080";
    out.fillRect(0, 0, size, size);
    out.drawImage(crop, Math.floor((size - width) / 2), Math.floor((size - height) / 2));
    const image = await new Promise(resolve => padded.toBlob(resolve, "image/png"));
    if (!image) throw new Error("마스크 적용 이미지를 만들지 못했습니다.");
    return image;
  } finally {
    bitmap.close();
  }
}
