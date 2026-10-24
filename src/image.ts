/**
 * Small browser image helpers. These are optional; the core pipeline works on
 * plain `ImageData`, so a worker or a canvas app can skip them.
 */

export type ImageSource =
  | ImageData
  | ImageBitmap
  | HTMLImageElement
  | HTMLCanvasElement
  | OffscreenCanvas
  | Blob
  | string;

function makeCanvas(w: number, h: number): {
  canvas: OffscreenCanvas | HTMLCanvasElement;
  ctx: CanvasRenderingContext2D | OffscreenCanvasRenderingContext2D;
} {
  if (typeof OffscreenCanvas !== 'undefined') {
    const canvas = new OffscreenCanvas(w, h);
    const ctx = canvas.getContext('2d', { willReadFrequently: true });
    if (!ctx) throw new Error('Could not create 2D canvas context');
    return { canvas, ctx };
  }
  const canvas = document.createElement('canvas');
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  if (!ctx) throw new Error('Could not create 2D canvas context');
  return { canvas, ctx };
}

/** Decode a URL, Blob, image, canvas, bitmap, or ImageData into ImageData. */
export async function toImageData(source: ImageSource): Promise<ImageData> {
  if (typeof ImageData !== 'undefined' && source instanceof ImageData) return source;

  let bitmap: ImageBitmap;
  if (typeof ImageBitmap !== 'undefined' && source instanceof ImageBitmap) {
    bitmap = source;
  } else if (source instanceof Blob) {
    bitmap = await createImageBitmap(source);
  } else if (typeof source === 'string') {
    const res = await fetch(source);
    if (!res.ok) throw new Error(`Could not load image: HTTP ${res.status}`);
    bitmap = await createImageBitmap(await res.blob());
  } else if (typeof HTMLImageElement !== 'undefined' && source instanceof HTMLImageElement) {
    if (!source.complete) {
      await new Promise<void>((resolve, reject) => {
        source.addEventListener('load', () => resolve(), { once: true });
        source.addEventListener('error', () => reject(new Error('Image failed to load')), {
          once: true,
        });
      });
    }
    bitmap = await createImageBitmap(source);
  } else if (typeof HTMLCanvasElement !== 'undefined' && source instanceof HTMLCanvasElement) {
    bitmap = await createImageBitmap(source);
  } else if (typeof OffscreenCanvas !== 'undefined' && source instanceof OffscreenCanvas) {
    bitmap = await createImageBitmap(source);
  } else {
    throw new Error('Unsupported image source');
  }

  const { ctx } = makeCanvas(bitmap.width, bitmap.height);
  ctx.drawImage(bitmap, 0, 0);
  const data = ctx.getImageData(0, 0, bitmap.width, bitmap.height);
  bitmap.close?.();
  return data;
}

/** Draw ImageData to a canvas element (handy for before/after previews). */
export function drawToCanvas(
  image: ImageData,
  canvas: HTMLCanvasElement | OffscreenCanvas,
): void {
  canvas.width = image.width;
  canvas.height = image.height;
  const ctx = canvas.getContext('2d') as
    | CanvasRenderingContext2D
    | OffscreenCanvasRenderingContext2D
    | null;
  if (!ctx) throw new Error('Could not create 2D canvas context');
  ctx.putImageData(image, 0, 0);
}

/** Encode ImageData as a PNG (or another canvas-supported type). */
export async function toBlob(image: ImageData, type = 'image/png'): Promise<Blob> {
  const { canvas } = makeCanvas(image.width, image.height);
  const ctx = canvas.getContext('2d') as
    | CanvasRenderingContext2D
    | OffscreenCanvasRenderingContext2D
    | null;
  if (!ctx) throw new Error('Could not create 2D canvas context');
  ctx.putImageData(image, 0, 0);
  if ('convertToBlob' in canvas) return canvas.convertToBlob({ type });
  return new Promise<Blob>((resolve, reject) => {
    (canvas as HTMLCanvasElement).toBlob(
      (blob) => (blob ? resolve(blob) : reject(new Error('Could not encode image'))),
      type,
    );
  });
}
