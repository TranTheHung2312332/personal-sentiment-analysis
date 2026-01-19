export default function CleanedText({ text }: { text: string }) {
  return (
    <section>
      <h3>Cleaned text</h3>
      <pre>{text}</pre>
    </section>
  )
}
