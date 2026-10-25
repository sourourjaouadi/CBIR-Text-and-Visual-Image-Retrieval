type Props = { keywords: string; setKeywords: (value: string) => void; onSearch: () => void; loading?: boolean };
export default function SearchBar({ keywords, setKeywords, onSearch, loading = false }: Props) {
  return <form className="searchbar" onSubmit={event => { event.preventDefault(); onSearch(); }}><input value={keywords} onChange={e => setKeywords(e.target.value)} placeholder="Search keywords, e.g. red blue"/><button disabled={loading}>{loading ? 'Searching…' : 'Search'}</button></form>;
}
