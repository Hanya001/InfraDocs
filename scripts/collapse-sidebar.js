(function () {
  const expandedDirectories = new Set();

  function directoryName(item) {
    return Array.from(item.childNodes)
      .filter(function (node) {
        return node.nodeName !== 'UL'
          && !(node.nodeType === Node.ELEMENT_NODE
            && node.classList.contains('sidebar-tree-toggle'));
      })
      .map(function (node) {
        return node.textContent;
      })
      .join('')
      .trim();
  }

  function directoryKey(item) {
    const names = [];
    let current = item;

    while (current && !current.classList.contains('sidebar-nav')) {
      if (current.tagName === 'LI') {
        names.unshift(directoryName(current));
      }
      current = current.parentElement;
    }

    return names.join('/');
  }

  function updateToggle(item, toggle, expanded) {
    const label = (expanded ? '收起 ' : '展开 ') + directoryName(item);

    item.classList.toggle('is-collapsed', !expanded);
    toggle.setAttribute('aria-expanded', String(expanded));
    toggle.setAttribute('aria-label', label);
    toggle.title = label;
  }

  function initialiseCollapsibleSidebar() {
    const items = document.querySelectorAll('.sidebar-nav li');

    items.forEach(function (item) {
      if (item.classList.contains('has-children')) {
        return;
      }

      const childList = Array.from(item.children).find(function (child) {
        return child.tagName === 'UL';
      });

      if (!childList) {
        return;
      }

      const key = directoryKey(item);
      const toggle = document.createElement('button');
      toggle.type = 'button';
      toggle.className = 'sidebar-tree-toggle';

      item.classList.add('has-children');
      item.dataset.directoryKey = key;
      item.insertBefore(toggle, item.firstChild);
      updateToggle(item, toggle, expandedDirectories.has(key));
    });
  }

  document.addEventListener('click', function (event) {
    const toggle = event.target.closest('.sidebar-tree-toggle');

    if (!toggle) {
      return;
    }

    const item = toggle.parentElement;
    const expanded = item.classList.contains('is-collapsed');
    const key = item.dataset.directoryKey;

    if (expanded) {
      expandedDirectories.add(key);
    } else {
      expandedDirectories.delete(key);
    }

    updateToggle(item, toggle, expanded);
  });

  window.$docsify.plugins = (window.$docsify.plugins || []).concat(function (hook) {
    hook.doneEach(initialiseCollapsibleSidebar);
  });
})();
