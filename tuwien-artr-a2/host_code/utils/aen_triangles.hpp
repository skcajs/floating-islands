#pragma once

namespace aen_triangles
{
	// Some structs which might prove helpful for the implementation of PN/AEN triangles:
	class vertex
	{
		friend struct std::hash<aen_triangles::vertex>;
	public:
		vertex() noexcept : m_index(0), m_pos(glm::vec3(0)) {}
		vertex(uint32_t index, const glm::vec3& pos) noexcept : m_index(index), m_pos(pos) {}
		vertex(vertex&&) noexcept = default;
		vertex(const vertex&) noexcept = default;
		vertex& operator=(vertex&&) noexcept = default;
		vertex& operator=(const vertex&) noexcept = default;
		bool operator==(const vertex& other) const { return m_index == other.m_index || m_pos == other.m_pos; }
		uint32_t index() const { return m_index; }
		const glm::vec3& position() const { return m_pos; }
	private:
		uint32_t m_index;
		glm::vec3 m_pos;
	};

	/*! Stores the start and end vertex of an edge. */
	class edge
	{
		friend struct std::hash<aen_triangles::edge>;
	public:
		edge(vertex& origin, vertex& destination) noexcept : m_origin(origin), m_destination(destination) {}
		edge(edge&&) noexcept = default;
		edge(const edge&) noexcept = default;
		edge& operator=(edge&&) noexcept = default;
		edge& operator=(const edge&) noexcept = default;
		bool operator==(const edge& other) const { return m_origin == other.m_origin && m_destination == other.m_destination; }
		const vertex& origin() const { return m_origin; }
		const vertex& destination() const { return m_destination; }
	private:
		vertex m_origin;
		vertex m_destination;
	};
}

namespace std
{
	template<> struct hash<aen_triangles::vertex>
	{
		size_t operator()(aen_triangles::vertex const& v) const noexcept
		{
			auto const h1(hash<float>{}(v.m_pos.x));
			auto const h2(hash<float>{}(v.m_pos.y));
			auto const h3(hash<float>{}(v.m_pos.z));
			return ((h1 ^ (h2 << 1)) >> 1) ^ (h3 << 1);
		}
	};

	template<> struct hash<aen_triangles::edge>
	{
		size_t operator()(aen_triangles::edge const& e) const noexcept
		{
			size_t const h1(hash<aen_triangles::vertex>{}(e.m_origin));
			size_t const h2(hash<aen_triangles::vertex>{}(e.m_destination));
			return h1 ^ (h2 << 1);
		}
	};
}


