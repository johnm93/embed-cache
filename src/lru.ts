export interface LruOptions<V> {
  /** Max number of entries. Default: unlimited. */
  maxEntries?: number;
  /** Max total bytes across all entries, per `sizeOf`. Default: unlimited. */
  maxBytes?: number;
  /** Cost of one value in bytes. Default: every value costs 0. */
  sizeOf?: (value: V) => number;
}

interface Entry<V> {
  value: V;
  bytes: number;
}

/**
 * A Map iterates in insertion order, so re-inserting a key on every touch
 * turns that order into recency order for free: oldest-first is
 * least-recently-used-first, which is exactly what eviction needs to walk.
 */
export class Lru<K, V> {
  private readonly map = new Map<K, Entry<V>>();
  private readonly maxEntries: number;
  private readonly maxBytes: number;
  private readonly sizeOf: (value: V) => number;
  private totalBytes = 0;

  constructor(options: LruOptions<V> = {}) {
    const { maxEntries = Infinity, maxBytes = Infinity, sizeOf = () => 0 } = options;
    if (maxEntries <= 0) throw new RangeError('maxEntries must be positive');
    if (maxBytes <= 0) throw new RangeError('maxBytes must be positive');
    this.maxEntries = maxEntries;
    this.maxBytes = maxBytes;
    this.sizeOf = sizeOf;
  }

  get size(): number {
    return this.map.size;
  }

  get bytes(): number {
    return this.totalBytes;
  }

  has(key: K): boolean {
    return this.map.has(key);
  }

  /** Read a value and mark it most-recently-used. */
  get(key: K): V | undefined {
    const entry = this.map.get(key);
    if (!entry) return undefined;
    this.map.delete(key);
    this.map.set(key, entry);
    return entry.value;
  }

  /** Read a value without affecting recency. */
  peek(key: K): V | undefined {
    return this.map.get(key)?.value;
  }

  set(key: K, value: V): void {
    const existing = this.map.get(key);
    if (existing) {
      this.totalBytes -= existing.bytes;
      this.map.delete(key);
    }
    const bytes = this.sizeOf(value);
    this.map.set(key, { value, bytes });
    this.totalBytes += bytes;
    this.evict();
  }

  delete(key: K): boolean {
    const entry = this.map.get(key);
    if (!entry) return false;
    this.totalBytes -= entry.bytes;
    this.map.delete(key);
    return true;
  }

  clear(): void {
    this.map.clear();
    this.totalBytes = 0;
  }

  keys(): IterableIterator<K> {
    return this.map.keys();
  }

  /**
   * Evict oldest-first until both budgets are met. A single entry larger
   * than maxBytes is still kept once it is the only one left, otherwise a
   * cache with a low byte budget could never hold anything at all.
   */
  private evict(): void {
    for (const [key, entry] of this.map) {
      if (this.map.size <= 1) break;
      if (this.map.size <= this.maxEntries && this.totalBytes <= this.maxBytes) break;
      this.map.delete(key);
      this.totalBytes -= entry.bytes;
    }
  }
}
