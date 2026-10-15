import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "PickFoot | Football Intelligence",
  description: "축구 데이터와 모델 추정 확률을 한 곳에서 분석하는 PickFoot",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="ko">
      <body>{children}</body>
    </html>
  );
}
