document.addEventListener('DOMContentLoaded', () => {
  const form = document.getElementById('predictionForm');
  const resultContainer = document.getElementById('resultContainer');

  const sepal1Value = document.getElementById('sepal1Value');
  const sepal2Value = document.getElementById('sepal2Value');
  const petal1Value = document.getElementById('petal1Value');
  const petal2Value = document.getElementById('petal2Value');

  form.sepal1.addEventListener('input', e => sepal1Value.textContent = e.target.value);
  form.sepal2.addEventListener('input', e => sepal2Value.textContent = e.target.value);
  form.petal1.addEventListener('input', e => petal1Value.textContent = e.target.value);
  form.petal2.addEventListener('input', e => petal2Value.textContent = e.target.value);

  form.addEventListener('submit', async (event) => {
    event.preventDefault();

    const data = {
      sepal: [parseFloat(form.sepal1.value), parseFloat(form.sepal2.value)],
      petal: [parseFloat(form.petal1.value), parseFloat(form.petal2.value)]
    };

    try {
      const response = await fetch('/', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(data)
      });

      if (!response.ok) throw new Error(`Server error: ${response.statusText}`);

      const predictions = await response.json();

      // پیدا کردن همه کلاس‌ها (کلیدهای مشترک بین همه الگوریتم‌ها)
      const allClasses = new Set();
      for (const result of Object.values(predictions)) {
        Object.keys(result).forEach(k => allClasses.add(k));
      }
      const classes = Array.from(allClasses);

      // ساخت جدول
      let table = '<table class="table table-bordered table-striped">';
      table += `<thead><tr><th>Algorithm</th>${classes.map(cls => `<th>${cls}</th>`).join('')}</tr></thead>`;
      table += '<tbody>';

      for (const [modelName, result] of Object.entries(predictions)) {
        const rowValues = classes.map(cls => parseFloat(result[cls]) || 0);
        const max = Math.max(...rowValues);
        const min = Math.min(...rowValues);

        const row = rowValues.map(val => {
          const rounded = val.toFixed(3);
          let colorClass = '';
          if (val === max) colorClass = 'table-success';
          else if (val === min) colorClass = 'table-danger';
          else colorClass = 'table-warning';
          return `<td class="${colorClass} text-center">${rounded}</td>`;
        });

        table += `<tr><td><strong>${modelName}</strong></td>${row.join('')}</tr>`;
      }

      table += '</tbody></table>';
      resultContainer.innerHTML = table;

    } catch (error) {
      resultContainer.innerHTML = `
        <div class="alert alert-danger" role="alert">
          Error: ${error.message}
        </div>
      `;
    }
  });
});
