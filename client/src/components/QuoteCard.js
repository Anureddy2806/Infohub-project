import React, { useEffect, useState } from 'react';

function QuoteCard() {
  const [quote, setQuote] = useState(undefined);

  useEffect(() => {
    const base = process.env.REACT_APP_API_URL;
   
    console.log('✅ API base:', base);

    const url = `${base}/api/quote`;
    console.log('🌐 Fetching from:', url);

    fetch(url)
      .then(res => {
        console.log('📦 Response status:', res.status);
        return res.json();
      })
      .then(data => {
        console.log('📨 Quote data received:', data);
        setQuote(data.quote);
      })
      .catch(err => console.error('❌ Quote fetch error:', err));
  }, []);

  return (
    <div className="module-card quote">
      <h2>Quote</h2>
      {quote !== undefined ? (
        <p>"{quote}"</p>
      ) : (
        <p>Loading quote...</p>
      )}
    </div>
  );
}

export default QuoteCard;