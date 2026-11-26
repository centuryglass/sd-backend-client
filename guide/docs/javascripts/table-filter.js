// Filters the table that follows each `input.guide-table-filter`, hiding rows whose text doesn't contain the query.
document.addEventListener('DOMContentLoaded', () => {
  document.querySelectorAll('input.guide-table-filter').forEach((input) => {
    let element = input.closest('p') || input;
    let table = null;
    while (element && !table) {
      element = element.nextElementSibling;
      if (element) {
        table = element.matches('table') ? element : element.querySelector('table');
      }
    }
    if (!table) {
      return;
    }
    const rows = Array.from(table.tBodies[0].rows);
    input.addEventListener('input', () => {
      const query = input.value.trim().toLowerCase();
      rows.forEach((row) => {
        row.hidden = query !== '' && !row.textContent.toLowerCase().includes(query);
      });
    });
  });
});
