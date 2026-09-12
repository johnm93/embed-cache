import { mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { EmbedCache } from '../src/embed-cache.ts';

const DIM = 8;

// Stands in for a real embeddings API: deterministic, no network, no cost.
function fakeEmbed(text: string): number[] {
  const vector = new Array(DIM).fill(0);
  for (let i = 0; i < text.length; i++) {
    vector[i % DIM] += text.charCodeAt(i);
  }
  return vector.map((v) => v / 100);
}

async function main(): Promise<void> {
  const dir = await mkdtemp(join(tmpdir(), 'embed-cache-example-'));
  try {
    const model = 'fake-embedding-v1';
    const texts = ['The quick brown fox', 'jumps over the lazy dog', 'Hello, world!', 'The quick brown fox'];

    let apiCalls = 0;
    let lastBatchSize = 0;
    const computeBatch = async (missing: string[]) => {
      apiCalls++;
      lastBatchSize = missing.length;
      return missing.map(fakeEmbed);
    };

    const cache = new EmbedCache({ dir, expectedDim: DIM });

    const first = await cache.getOrComputeMany(model, texts, computeBatch);
    console.log(`first pass:  ${first.length} vectors, ${apiCalls} api call(s) for ${lastBatchSize} text(s)`);

    const second = await cache.getOrComputeMany(model, texts, computeBatch);
    console.log(`second pass: ${second.length} vectors, ${apiCalls} api call(s) total`);

    const identical = first.every((v, i) => v.every((x, j) => x === second[i]![j]));
    console.log(`identical:   ${identical}`);

    // A fresh instance over the same directory stands in for restarting the process:
    // memory is empty, so this can only be answered from disk.
    const restarted = new EmbedCache({ dir, expectedDim: DIM });
    const afterRestart = await restarted.get(model, texts[0]!);
    console.log(`after restart: dim ${afterRestart!.length}, stats ${JSON.stringify(restarted.stats())}`);

    const stats = cache.stats();
    console.log(
      `hit rate ${(stats.hitRate * 100).toFixed(1)}%  ` +
        `(memory ${stats.memoryHits}, disk ${stats.diskHits}, miss ${stats.misses}, computed ${stats.computed})  ` +
        `${stats.entries} entries / ${stats.bytes} bytes in memory`,
    );
  } finally {
    await rm(dir, { recursive: true, force: true });
  }
}

main();
