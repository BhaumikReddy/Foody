import { Link, useLocation } from 'react-router-dom';

export default function Navbar() {
  const location = useLocation();

  const navLink = (to, label) => {
    const isActive = location.pathname === to;
    return (
      <Link
        to={to}
        className={`text-sm font-medium pb-0.5 transition-colors duration-150 ${
          isActive
            ? 'text-brand-500 border-b-2 border-brand-500'
            : 'text-gray-500 hover:text-gray-900'
        }`}
      >
        {label}
      </Link>
    );
  };

  return (
    <nav className="bg-white border-b border-gray-100 sticky top-0 z-50">
      <div className="max-w-6xl mx-auto px-4 h-14 flex items-center justify-between">
        <Link to="/" className="text-xl font-bold text-brand-500 tracking-tight">
          foody
        </Link>
        <div className="flex items-center gap-6">
          {navLink('/', 'Analyse')}
          {navLink('/dashboard', 'Dashboard')}
        </div>
      </div>
    </nav>
  );
}
