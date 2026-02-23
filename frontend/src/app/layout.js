import "./globals.css";

export const metadata = {
  title: "BibliTrace",
  description: "BibliTrace",
};

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
