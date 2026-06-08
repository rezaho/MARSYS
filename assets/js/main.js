// Main JavaScript file for Multi-Agent Reasoning Systems (MARSYS) documentation

// Add any custom JavaScript functionality here
document.addEventListener('DOMContentLoaded', function() {
    console.log('DOM loaded - MARSYS ready');

    // Smooth scrolling for internal links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });

    // Add glassmorphism effect on scroll
    const header = document.querySelector('.md-header');
    if (header) {
        window.addEventListener('scroll', function() {
            if (window.scrollY > 50) {
                header.style.backdropFilter = 'blur(25px)';
                header.style.boxShadow = '0 8px 32px rgba(0, 0, 0, 0.1)';
            } else {
                header.style.backdropFilter = 'blur(20px)';
                header.style.boxShadow = '0 4px 20px rgba(0, 0, 0, 0.1)';
            }
        });
    }

    // Enhance code blocks with copy functionality feedback
    const copyButtons = document.querySelectorAll('.md-clipboard');
    copyButtons.forEach(button => {
        button.addEventListener('click', function() {
            const icon = this.querySelector('.md-clipboard__message');
            if (icon) {
                icon.textContent = 'Copied!';
                setTimeout(() => {
                    icon.textContent = '';
                }, 2000);
            }
        });
    });
});

// Add loading animation for page transitions
document.addEventListener('turbo:load', function() {
    document.body.classList.add('loaded');
});

// Mermaid diagram initialization (if needed)
if (typeof mermaid !== 'undefined') {
    mermaid.initialize({ 
        startOnLoad: true,
        theme: 'default',
        themeVariables: {
            primaryColor: '#6366f1',
            primaryTextColor: '#1f2937',
            primaryBorderColor: '#e5e7eb',
            lineColor: '#9ca3af',
            secondaryColor: '#f3f4f6',
            tertiaryColor: '#fef3c7'
        }
    });
}

// Navigation forcing removed to allow natural MkDocs behavior




// Sidebar navigation overrides removed to allow natural MkDocs behavior

// Add loading animation for page transitions
document.addEventListener('turbo:load', function() {
    document.body.classList.add('loaded');
});

// Mermaid diagram initialization (if needed)
if (typeof mermaid !== 'undefined') {
    mermaid.initialize({ 
        startOnLoad: true,
        theme: 'default',
        themeVariables: {
            primaryColor: '#6366f1',
            primaryTextColor: '#1f2937',
            primaryBorderColor: '#e5e7eb',
            lineColor: '#9ca3af',
            secondaryColor: '#f3f4f6',
            tertiaryColor: '#fef3c7'
        }
    });
}
