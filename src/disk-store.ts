import { mkdir, readFile, rename, rm, unlink, writeFile } from 'node:fs/promises';
import { randomBytes } from 'node:crypto';
import { dirname, join } from 'node:path';
import { decodeVector, encodeVector } from './codec.js';
import { keyToSegments } from './key.js';
import type { VectorInput } from './types.js';

export interface DiskGetResult {
  /** The decoded vector, or null on a miss or a corrupt file. */
  vector: Float32Array | null;
  /** True when a file existed but failed its header, checksum, or dim check. */
  corrupt: boolean;
}

/**
 * One file per key at `<dir>/<key[0:2]>/<key[2:4]>/<key[4:]>.vec` (see
 * README for the exact byte layout). Writes go to a temp file in the same
 * shard directory and are renamed into place, so a reader never sees a
 * partially written file: `rename` is atomic within a filesystem, and the
 * temp file lives next to its target so the two are guaranteed to share one.
 */
export class DiskStore {
  readonly dir: string;

  constructor(dir: string) {
    if (typeof dir !== 'string' || dir.length === 0) {
      throw new TypeError('dir must be a non-empty string');
    }
    this.dir = dir;
  }

  private pathFor(key: string): string {
    return join(this.dir, ...keyToSegments(key));
  }

  async get(key: string, expectedDim?: number | null): Promise<DiskGetResult> {
    let buf: Buffer;
    try {
      buf = await readFile(this.pathFor(key));
    } catch (err: any) {
      if (err.code === 'ENOENT') return { vector: null, corrupt: false };
      throw err;
    }

    const decoded = decodeVector(buf, expectedDim);
    if (!decoded) {
      // A file that fails to decode is never served -- drop it so the next
      // read is a plain miss instead of repeating the same failure.
      await this.delete(key).catch(() => {});
      return { vector: null, corrupt: true };
    }
    return { vector: decoded.vector, corrupt: false };
  }

  async set(key: string, input: VectorInput): Promise<void> {
    const path = this.pathFor(key);
    const dir = dirname(path);
    await mkdir(dir, { recursive: true });

    const tmp = join(dir, `.${key}.${randomBytes(6).toString('hex')}.tmp`);
    const buf = encodeVector(input);
    try {
      await writeFile(tmp, buf);
      await rename(tmp, path);
    } catch (err) {
      await unlink(tmp).catch(() => {});
      throw err;
    }
  }

  async delete(key: string): Promise<boolean> {
    try {
      await unlink(this.pathFor(key));
      return true;
    } catch (err: any) {
      if (err.code === 'ENOENT') return false;
      throw err;
    }
  }

  /** Removes every stored vector. The store is still usable afterward. */
  async clear(): Promise<void> {
    await rm(this.dir, { recursive: true, force: true });
  }
}
