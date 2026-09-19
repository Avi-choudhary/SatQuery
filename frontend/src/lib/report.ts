import { ChatMessage, Session } from './types';

export function printReport(title: string, messages: ChatMessage[]) {
  const printWindow = window.open('', '_blank');
  if (!printWindow) {
    alert('Please allow popups to generate the report.');
    return;
  }

  // Clone the styles from the current document
  const styles = Array.from(document.head.querySelectorAll('style, link[rel="stylesheet"]'))
    .map((el) => el.outerHTML)
    .join('\n');

  // Format messages
  const messagesHtml = messages.map(msg => {
    const isUser = msg.role === 'user';
    const roleName = isUser ? 'User' : 'SatQuery AI';
    const roleColor = isUser ? 'var(--color-accent)' : 'var(--color-ink)';
    
    // We can parse markdown or just leave it as text for simplicity, 
    // but preserving pre-wrap handles newlines.
    
    let evidenceHtml = '';
    if (msg.evidence && msg.evidence.featureCount > 0) {
      evidenceHtml = `
        <div style="margin-top: 8px; padding: 8px 12px; background: rgba(16, 185, 129, 0.1); border: 1px solid rgba(16, 185, 129, 0.2); border-radius: 6px; font-family: monospace; font-size: 12px; color: #10b981;">
          <strong>Evidence:</strong> ${msg.evidence.featureCount} region(s) detected on the map
          ${msg.evidence.areaHa ? `&middot; ${msg.evidence.areaHa.toLocaleString()} ha` : ''}
        </div>
      `;
    }

    return `
      <div style="margin-bottom: 24px; padding: 16px; border-radius: 12px; ${isUser ? 'background: rgba(99, 102, 241, 0.05); border: 1px solid rgba(99, 102, 241, 0.15);' : 'background: #f9fafb; border: 1px solid #e5e7eb;'}">
        <div style="font-weight: 600; font-size: 14px; margin-bottom: 8px; color: ${isUser ? '#4f46e5' : '#111827'};">${roleName}</div>
        <div style="white-space: pre-wrap; font-size: 14px; line-height: 1.6; color: #111827;">${msg.content}</div>
        ${evidenceHtml}
      </div>
    `;
  }).join('');

  const html = `
    <!DOCTYPE html>
    <html lang="en">
      <head>
        <meta charset="UTF-8">
        <title>SatQuery Report - ${title || 'Chat Session'}</title>
        ${styles}
        <style>
          :root {
            /* Force light theme for printing */
            --color-ink: #111827;
            --color-ground: #ffffff;
            --color-surface: #f9fafb;
            --color-line: #e5e7eb;
          }
          body { 
            padding: 40px; 
            background: white !important; 
            color: #111827 !important; 
            font-family: ui-sans-serif, system-ui, sans-serif;
          }
          @media print {
            body { padding: 0; }
            * { -webkit-print-color-adjust: exact !important; print-color-adjust: exact !important; }
          }
          .header {
            margin-bottom: 32px;
            padding-bottom: 16px;
            border-bottom: 2px solid #e5e7eb;
          }
          .title {
            font-size: 24px;
            font-weight: bold;
            margin: 0 0 8px 0;
            color: #111827;
          }
          .meta {
            font-size: 13px;
            color: #6b7280;
            margin: 0;
          }
          .footer {
            margin-top: 48px;
            padding-top: 16px;
            border-top: 1px solid #e5e7eb;
            font-size: 12px;
            color: #9ca3af;
            text-align: center;
          }
        </style>
      </head>
      <body>
        <div style="max-w: 800px; margin: 0 auto;">
          <div class="header">
            <h1 class="title">SatQuery AI Analysis Report</h1>
            <p class="meta">Session: ${title || 'Untitled Session'}</p>
            <p class="meta">Generated on: ${new Date().toLocaleString()}</p>
          </div>
          
          <div class="messages">
            ${messagesHtml}
          </div>
          
          <div class="footer">
            Generated securely by SatQuery AI
          </div>
        </div>
        
        <script>
          // Wait for custom fonts to load before printing
          window.onload = () => {
            setTimeout(() => {
              window.print();
            }, 500);
          };
        </script>
      </body>
    </html>
  `;
  
  printWindow.document.write(html);
  printWindow.document.close();
}
