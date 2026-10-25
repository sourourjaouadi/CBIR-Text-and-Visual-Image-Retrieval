import type { ChangeEvent } from 'react';

type Props = {
  imageId: number | null;
  imageUrl: string | null;
  keywords: string[];
  queryLoading: boolean;
  onImageIdChange: (id: number | null) => void;
  onLoadImage: () => void;
  onUpload: (file: File | null) => void;
};

export default function QueryImagePanel({ imageId, imageUrl, keywords, queryLoading,
  onImageIdChange, onLoadImage, onUpload }: Props) {
  function selectFile(event: ChangeEvent<HTMLInputElement>) {
    onUpload(event.target.files?.[0] ?? null);
    event.target.value = '';
  }
  return <section className="query-panel">
    <div className="query-preview">
      {imageUrl ? <img src={imageUrl} alt="Current query"/> : <div className="image-placeholder">Query<br/>image</div>}
    </div>
    <div className="query-info">
      <div className="query-title-row"><div><p className="eyebrow">VISUAL QUERY</p><h2>{imageId ? `Image #${imageId}` : imageUrl ? 'Uploaded image' : 'Choose an image'}</h2></div>
        <label className="upload-button">Upload image<input type="file" accept="image/*" onChange={selectFile}/></label></div>
      <div className="id-load-row"><label htmlFor="query-id">Image ID</label><input id="query-id" type="number" min="1" value={imageId ?? ''} onChange={e => onImageIdChange(e.target.value ? Number(e.target.value) : null)} onKeyDown={e => { if (e.key === 'Enter') onLoadImage(); }}/><button className="subtle-button" onClick={onLoadImage} disabled={!imageId || queryLoading}>{queryLoading ? 'Loading…' : 'Load image'}</button></div>
      <div className="keyword-chips"><span className="muted-label">Keywords</span>{keywords.length ? keywords.map(word => <span className="chip" key={word}>{word}</span>) : <span className="muted-copy">No keywords loaded</span>}</div>
      <p className="panel-hint">Click any result to use it as the next visual query.</p>
    </div>
  </section>;
}
