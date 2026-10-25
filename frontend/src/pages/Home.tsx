import { useEffect, useMemo, useState } from 'react';
import { descriptors, getImage, search, searchUpload, type Descriptor, type ImageInfo, type Result } from '../api/client';
import SearchBar from '../components/SearchBar';
import DescriptorPicker, { type DescriptorInfo } from '../components/DescriptorPicker';
import QueryImagePanel from '../components/QueryImagePanel';
import ResultsGrid from '../components/ResultsGrid';

const PAGE_SIZE = 12;

export default function Home() {
  const [dataset, setDataset] = useState('objects');
  const [keywords, setKeywords] = useState('');
  const [mode, setMode] = useState('and');
  const [queryId, setQueryId] = useState<number | null>(null);
  const [queryImage, setQueryImage] = useState<ImageInfo | null>(null);
  const [queryUrl, setQueryUrl] = useState<string | null>(null);
  const [uploadPreview, setUploadPreview] = useState<string | null>(null);
  const [uploadFile, setUploadFile] = useState<File | null>(null);
  const [available, setAvailable] = useState<DescriptorInfo[]>([]);
  const [descriptorLoading, setDescriptorLoading] = useState(true);
  const [descriptorError, setDescriptorError] = useState('');
  const [selected, setSelected] = useState<Descriptor[]>([]);
  const [results, setResults] = useState<Result[]>([]);
  const [elapsed, setElapsed] = useState<number | null>(null);
  const [k, setK] = useState(24);
  const [page, setPage] = useState(1);
  const [loading, setLoading] = useState(false);
  const [queryLoading, setQueryLoading] = useState(false);
  const [error, setError] = useState('');

  useEffect(() => {
    let active = true;
    setDescriptorLoading(true);
    setDescriptorError('');
    descriptors(dataset).then(data => {
      if (!active) return;
      const names = data.descriptors;
      setAvailable(names);
      setSelected(current => current.filter(item => names.some(info => info.name === item.name)));
    }).catch(err => {
      if (active) { setAvailable([]); setDescriptorError(String(err)); }
    }).finally(() => { if (active) setDescriptorLoading(false); });
    return () => { active = false; };
  }, [dataset]);

  const visibleResults = useMemo(() => {
    const ordered = [...results].sort((a, b) => a.distance - b.distance || a.image_id - b.image_id);
    const start = (page - 1) * PAGE_SIZE;
    return ordered.slice(start, start + PAGE_SIZE).map((item, index) => ({ ...item, rank: start + index + 1 }));
  }, [results, page]);
  const pageCount = Math.max(1, Math.ceil(results.length / PAGE_SIZE));
  useEffect(() => {
    if (!uploadFile) { setUploadPreview(null); return; }
    const url = URL.createObjectURL(uploadFile);
    setUploadPreview(url);
    return () => URL.revokeObjectURL(url);
  }, [uploadFile]);
  const displayedQueryUrl = uploadFile ? uploadPreview : queryUrl;

  async function runSearch(overrides: { imageId?: number | null; file?: File | null } = {}) {
    setLoading(true);
    setError('');
    setPage(1);
    const imageId = overrides.imageId !== undefined ? overrides.imageId : queryId;
    const file = overrides.file !== undefined ? overrides.file : uploadFile;
    const payload = {
      dataset,
      keywords: keywords.split(/[, ]+/).map(word => word.trim()).filter(Boolean),
      keyword_mode: mode,
      query_image_id: file ? null : imageId,
      descriptors: selected.filter(item => available.some(info => info.name === item.name)),
      k,
    };
    try {
      const response = file ? await searchUpload(payload, file) : await search(payload);
      setResults(response.results);
      setElapsed(response.took_ms);
      setQueryUrl(response.query_image_url);
      if (file) {
        setQueryImage({ image_id: 0, url: response.query_image_url ?? '', label: 'upload',
          keywords: payload.keywords });
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
      setResults([]);
    } finally { setLoading(false); }
  }

  async function loadQueryImage() {
    if (!queryId) { setQueryImage(null); setQueryUrl(null); return; }
    setQueryLoading(true);
    setError('');
    try {
      const image = await getImage(dataset, queryId);
      setQueryImage(image);
      setQueryUrl(image.url);
      setUploadFile(null);
    } catch (err) {
      setQueryImage(null);
      setQueryUrl(null);
      setError(err instanceof Error ? err.message : String(err));
    } finally { setQueryLoading(false); }
  }

  function onUpload(file: File | null) {
    setUploadFile(file);
    setQueryId(null);
    setQueryImage(file ? { image_id: 0, url: '', label: 'upload', keywords: [] } : null);
    setQueryUrl(null);
  }

  function chooseResult(result: Result) {
    setQueryId(result.image_id);
    setQueryImage({ image_id: result.image_id, url: result.url, label: null, keywords: result.keywords ?? [] });
    setQueryUrl(result.url);
    setUploadFile(null);
    void runSearch({ imageId: result.image_id, file: null });
  }

  function changeDataset(value: string) {
    setDataset(value);
    setResults([]); setQueryId(null); setQueryImage(null); setQueryUrl(null); setUploadFile(null);
  }

  return <main>
    <header className="page-header">
      <div><p className="eyebrow">CONTENT-BASED IMAGE RETRIEVAL</p><h1>Find images by meaning and appearance.</h1><p className="subtitle">Search your collection by keywords, visual features, or both.</p></div>
      <label className="dataset-select">Dataset<select value={dataset} onChange={event => changeDataset(event.target.value)}><option value="objects">Objects demo</option><option value="mpeg7_1m">MPEG-7 collection</option></select></label>
    </header>

    <section className="panel search-panel">
      <SearchBar keywords={keywords} setKeywords={setKeywords} onSearch={() => void runSearch()} loading={loading}/>
      <div className="controls">
        <label className="control-label">Keyword matching<select value={mode} onChange={event => setMode(event.target.value)}><option value="and">All terms (AND)</option><option value="or">Any term (OR)</option></select></label>
        <label className="control-label">Results<select value={k} onChange={event => setK(Number(event.target.value))}>{[12, 24, 48, 100, 250, 500].map(value => <option value={value} key={value}>{value}</option>)}</select></label>
      </div>
      <QueryImagePanel imageId={queryId} imageUrl={displayedQueryUrl} keywords={queryImage?.keywords ?? []}
        queryLoading={queryLoading} onImageIdChange={id => { setQueryId(id); setQueryImage(null); setQueryUrl(null); if (uploadFile) setUploadFile(null); }}
        onLoadImage={() => void loadQueryImage()} onUpload={onUpload}/>
      <div className="descriptor-section"><div className="section-heading"><h2>Visual descriptors</h2><span>{descriptorLoading ? 'Loading descriptors…' : `${selected.length} selected`}</span></div>
        {descriptorError ? <p className="inline-error">Could not load descriptors: {descriptorError}</p> : <DescriptorPicker names={available} value={selected} onChange={setSelected} disabled={descriptorLoading}/>}
        <p className="hint">Choose visual features and adjust their contribution to the ranking.</p>
      </div>
    </section>

    <section className="results-heading"><div><p className="eyebrow">MATCHES</p><h2>Results</h2></div><div className="result-summary">{loading ? <span>Searching…</span> : elapsed !== null ? <span>{results.length} images · {elapsed.toFixed(1)} ms</span> : null}</div></section>
    {error && <div className="error-banner" role="alert">{error}</div>}
    {loading ? <div className="loading-state"><span className="spinner"/>Finding visually similar images…</div> : <ResultsGrid results={visibleResults} onChoose={chooseResult}/>}
    {results.length > PAGE_SIZE && <nav className="pagination" aria-label="Results pages"><button onClick={() => setPage(current => Math.max(1, current - 1))} disabled={page <= 1}>Previous</button><span>Page {page} of {pageCount}</span><button onClick={() => setPage(current => Math.min(pageCount, current + 1))} disabled={page >= pageCount}>Next</button></nav>}
  </main>;
}
