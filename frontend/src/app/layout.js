import "./globals.css";

export const metadata = {
  title: "BibliTrace",
  description: "Претрага појмова и тема у Новом завету",
};

export default function RootLayout({ children }) {
  return (
    <html lang="sr-RS">
      <body>{children}</body>
    </html>
  );
}
